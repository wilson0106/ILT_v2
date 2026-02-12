import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import plotly.graph_objects as go

import webbrowser
import pandas as pd
import numpy as np
import json
import pickle
import sys
from collections import Counter

from os import listdir, getcwd, mkdir, makedirs
from os.path import join, isdir, exists, isfile
from shutil import copy2

from update_conf import update_csv_file

# --- clustering deps (scipy) ---
try:
    from scipy.cluster.hierarchy import fcluster, dendrogram
except Exception:
    fcluster = None
    dendrogram = None


# ======================================================================================
# GLOBALS (Dash server-side state)
# ======================================================================================

projects_list = []
classes_per_batch = {}
loadbar_classes = 0
loadbar_save_dataset = 0
current_classes_list = []
previous_slider_values = []
previous_var_slider_values = []

# Cache de artefatos e do último "mapa name -> cluster_id" calculado
# (não vai para o browser; evita explodir dcc.Store com milhões de chaves)
CLUSTER_ARTIFACT_CACHE = {}     # key: (project, class) -> dict(names=np.array, linkage=np.array, dist_max=float)
CLUSTER_ASSIGNMENT_CACHE = {}   # key: (project, class, cut) -> dict(name -> cluster_id)

# cut por classe (server-side)
# key: (project, class) -> cut01 float
CLUSTER_CUTS = {}


# ======================================================================================
# Helpers
# ======================================================================================

def get_projects_list():
    global projects_list
    projects_folder = join(getcwd(), 'main', 'assets')
    projects_list = [f for f in listdir(projects_folder) if isdir(join(projects_folder, f))]
    projects_list.sort()


def _ensure_manual_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige casos em que manual_label existe mas veio como booleano (True/False),
    ou veio faltando.
    Estratégia:
      - se manual_label não existir: usa preds1 (ou pred) como label
      - se manual_label existir mas for booleano / {True,False,"True","False"}: usa preds1 (ou pred)
      - caso contrário: força string
    """
    if 'manual_label' not in df.columns:
        if 'preds1' in df.columns:
            df['manual_label'] = df['preds1'].astype(str)
        elif 'pred' in df.columns:
            df['manual_label'] = df['pred'].astype(str)
        else:
            df['manual_label'] = "UNK"
        return df

    s = df['manual_label']
    try:
        vals = set(pd.Series(s).dropna().unique().tolist())
    except Exception:
        vals = set()

    is_boolish = False
    if pd.api.types.is_bool_dtype(s):
        is_boolish = True
    else:
        # cobre caso string "True"/"False"
        if vals and vals.issubset({True, False, "True", "False"}):
            is_boolish = True

    if is_boolish:
        if 'preds1' in df.columns:
            df['manual_label'] = df['preds1'].astype(str)
        elif 'pred' in df.columns:
            df['manual_label'] = df['pred'].astype(str)
        else:
            df['manual_label'] = s.astype(str)
    else:
        df['manual_label'] = s.astype(str)

    return df


def _clusters_folder(project_name: str) -> str:
    return join(getcwd(), 'main', 'assets', project_name, 'clusters')


def _sanitize_filename(x: str) -> str:
    return x.replace("/", "_").replace("\\", "_").strip()


def _find_cluster_artifact(project_name: str, class_label: str):
    """
    Procura arquivo de cluster em:
      main/assets/<project>/clusters/<class_label>.npz
      main/assets/<project>/clusters/<sanitized>.npz
      (também aceita .pkl)
    """
    folder = _clusters_folder(project_name)
    if not isdir(folder):
        return None

    candidates = [
        join(folder, f"{class_label}.npz"),
        join(folder, f"{_sanitize_filename(class_label)}.npz"),
        join(folder, f"{class_label}.pkl"),
        join(folder, f"{_sanitize_filename(class_label)}.pkl"),
    ]

    for p in candidates:
        if isfile(p):
            return p
    return None


def _load_cluster_artifact(project_name: str, class_label: str):
    key = (project_name, class_label)
    if key in CLUSTER_ARTIFACT_CACHE:
        return CLUSTER_ARTIFACT_CACHE[key]

    path = _find_cluster_artifact(project_name, class_label)
    if path is None:
        return None

    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)

        # names sempre string
        names = data["names"]
        try:
            names = np.asarray(names).astype(str)
        except Exception:
            names = np.array([str(x) for x in names])

        # garante float64 (double) para o scipy
        Z = np.asarray(data["linkage"], dtype=np.float64)

        # dist_max robusto (pode vir como array 0-d)
        if "dist_max" in data:
            try:
                dist_max = float(np.asarray(data["dist_max"]).reshape(-1)[0])
            except Exception:
                dist_max = float(data["dist_max"])
        else:
            dist_max = float(np.max(Z[:, 2])) if Z.size else 1.0

        artifact = {"names": names, "linkage": Z, "dist_max": dist_max, "path": path}

    else:
        with open(path, "rb") as f:
            d = pickle.load(f)

        # names sempre string
        names = d["names"]
        try:
            names = np.asarray(names).astype(str)
        except Exception:
            names = np.array([str(x) for x in names])

        # garante float64 (double) para o scipy
        Z = np.asarray(d["linkage"], dtype=np.float64)

        dist_max = float(d.get("dist_max", float(np.max(Z[:, 2])) if Z.size else 1.0))
        artifact = {"names": names, "linkage": Z, "dist_max": dist_max, "path": path}

    CLUSTER_ARTIFACT_CACHE[key] = artifact
    return artifact


def _build_dendrogram_figure(Z: np.ndarray, cut_dist: float, max_leaves: int = 800, hard_cap: int = 50000):
    """
    Dendrograma com:
      - acima da linha de corte: preto
      - abaixo da linha de corte: 1 cor por grupo (definido por color_threshold)

    Ajustes para evitar travamentos:
      - NÃO usa truncate_mode (foi a origem do RecursionError em casos grandes)
      - aumenta recursionlimit dinamicamente apenas durante o cálculo
      - se n for absurdo (hard_cap), não tenta desenhar (mas o cut pode ser salvo pelo caller)
    """
    fig = go.Figure()
    if dendrogram is None:
        fig.add_annotation(
            text="scipy não disponível: não é possível desenhar dendrograma.",
            showarrow=False
        )
        return fig

    if Z is None or (not hasattr(Z, "size")) or (Z.size == 0):
        fig.add_annotation(text="Sem dados suficientes para dendrograma.", showarrow=False)
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    # scipy quer float64
    Z = np.asarray(Z, dtype=np.float64)

    n = Z.shape[0] + 1
    if n <= 1:
        fig.add_annotation(text="Sem dados suficientes para dendrograma.", showarrow=False)
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    # proteção contra artefatos gigantes (ex.: linkage full de 800k)
    if n > hard_cap:
        fig.add_annotation(
            text=(f"Dendrograma não desenhado (N={n} > hard_cap={hard_cap}). "
                  f"Para classes enormes, gere artefato reduzido (centroid) com N bem menor."),
            showarrow=False
        )
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    # aumenta recursionlimit só durante a chamada (evita RecursionError)
    old_limit = sys.getrecursionlimit()
    need_limit = max(old_limit, int(5 * n + 2000))
    if need_limit > old_limit:
        sys.setrecursionlimit(need_limit)

    try:
        d = dendrogram(Z, no_plot=True, color_threshold=cut_dist)
    finally:
        if sys.getrecursionlimit() != old_limit:
            sys.setrecursionlimit(old_limit)

    icoord = d["icoord"]
    dcoord = d["dcoord"]
    color_list = d.get("color_list", ["C0"] * len(icoord))

    # palette "C0..C9" (matplotlib default)
    mpl_cycle = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    def _map_mpl_color(c: str) -> str:
        if isinstance(c, str) and len(c) == 2 and c[0] == "C" and c[1].isdigit():
            idx = int(c[1])
            return mpl_cycle[idx % len(mpl_cycle)]
        return c if isinstance(c, str) else "#1f77b4"

    traces = {}  # color -> {"x":[], "y":[]}

    for xs, ys, col in zip(icoord, dcoord, color_list):
        # acima (ou tocando) o corte -> preto
        is_above = (max(ys) >= cut_dist)
        key_color = "black" if is_above else _map_mpl_color(col)

        if key_color not in traces:
            traces[key_color] = {"x": [], "y": []}

        traces[key_color]["x"].extend([xs[0], xs[1], xs[2], xs[3], None])
        traces[key_color]["y"].extend([ys[0], ys[1], ys[2], ys[3], None])

    for c, xy in traces.items():
        fig.add_trace(go.Scatter(
            x=xy["x"],
            y=xy["y"],
            mode="lines",
            line=dict(color=c, width=1.5),
            hoverinfo="skip",
            showlegend=False
        ))

    # linha do corte
    fig.add_hline(y=cut_dist, line_dash="dash", line_color="black", opacity=0.9)

    # Mantém layout igual ao original (sem truncation)
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title="distance", showgrid=False, zeroline=False),
    )
    return fig





def _cuts_for_project(project_name: str) -> dict:
    out = {}
    for (p, c), cut in CLUSTER_CUTS.items():
        if p == project_name:
            out[c] = float(cut)
    return out


def _format_cuts_lines(project_name: str) -> str:
    cuts = _cuts_for_project(project_name)
    if not cuts:
        return "(nenhum cut definido ainda)"
    lines = []
    for c in sorted(cuts.keys()):
        lines.append(f"  - {c}: {cuts[c]:.2f}")
    return "\n".join(lines)


def _cluster_checklist_options(project_name: str, classes_list: list) -> list:
    cuts = _cuts_for_project(project_name) if project_name else {}
    opts = []
    for c in classes_list:
        cut_txt = f" | cut={cuts[c]:.2f}" if c in cuts else ""
        has_art = (_find_cluster_artifact(project_name, c) is not None) if project_name else False
        art_txt = " ✅" if has_art else " ⚠️(sem artefato)"
        opts.append({"label": f"{c}{cut_txt}{art_txt}", "value": c})
    return opts


def update_classes_project(batches_list, project_name):
    """
    Monta lista de classes (com contagens filtradas e total) e
    inclui "cut=XX" quando existir cut salvo para aquela classe.
    """
    global classes_per_batch

    temp_classes_project = {}
    temp_classes_project_total = {}

    for file in batches_list:
        if file in classes_per_batch:
            dict_temp = classes_per_batch[file]
            for k in dict_temp.keys():
                temp_classes_project[k] = temp_classes_project.get(k, 0) + dict_temp[k]

    batches_folder = join(getcwd(), 'main', 'assets', project_name, 'dataframes')
    for file in batches_list:
        file_path = join(batches_folder, file)
        df_temp = pd.read_csv(file_path)
        df_temp = _ensure_manual_label(df_temp)
        dict_temp = dict(df_temp['manual_label'].value_counts())
        for k in dict_temp.keys():
            temp_classes_project_total[k] = temp_classes_project_total.get(k, 0) + dict_temp[k]

    temp_keys = list(temp_classes_project.keys())
    temp_keys = [str(k) for k in temp_keys]
    temp_keys.sort()

    counts = [temp_classes_project.get(k, 0) for k in temp_keys]
    total_counts = [temp_classes_project_total.get(k, 0) for k in temp_keys]

    cuts = _cuts_for_project(project_name)

    classes_text = []
    for k, c, t in zip(temp_keys, counts, total_counts):
        cut_txt = f" | cut={cuts[k]:.2f}" if k in cuts else ""
        classes_text.append({'label': f' {k} ({c}/{t}){cut_txt}', 'value': 'enable'})

    classes_list = [k for k in temp_keys]
    return classes_text, classes_list


def get_classes(batches_folder, batches_list):
    classes_list = []
    for file in batches_list:
        file_path = join(batches_folder, file)
        df_temp = pd.read_csv(file_path)
        df_temp = _ensure_manual_label(df_temp)
        classes_list.extend(df_temp['manual_label'].astype(str).unique())
    classes_list = list(set(classes_list))
    classes_list.sort()
    return classes_list


def get_histogram_data_metric(batches_folder, batches_list, classes_list, metric_col):
    global loadbar_classes
    hist_data = {label: [] for label in classes_list}

    for i, file in enumerate(batches_list):
        file_path = join(batches_folder, file)
        df_temp = pd.read_csv(file_path)
        df_temp = _ensure_manual_label(df_temp)

        if metric_col not in df_temp.columns:
            if metric_col == 'manual_conf':
                try:
                    df_temp = update_csv_file(file_path)
                except Exception:
                    df_temp[metric_col] = 0.0
            else:
                df_temp[metric_col] = 0.0

        df_temp[metric_col] = pd.to_numeric(df_temp[metric_col], errors='coerce').fillna(0.0)

        for label in classes_list:
            vals = df_temp[df_temp['manual_label'] == label][metric_col].tolist()
            hist_data[label].extend(vals)

        loadbar_classes = 20 + 60 * i / max(1, len(batches_list))

    return hist_data


def get_classes_per_batch(batches_folder, batches_list, classes_list,
                         slider_values_conf=None, slider_values_var=None, changed_slider_index=None):
    global classes_per_batch, loadbar_classes

    if changed_slider_index is None:
        classes_per_batch = {}

    default_conf = [0.0, 1.0]
    default_var = [0.0, 1.0]

    for file in batches_list:
        file_path = join(batches_folder, file)
        df_temp = pd.read_csv(file_path)
        df_temp = _ensure_manual_label(df_temp)

        if 'manual_conf' not in df_temp.columns:
            try:
                df_temp = update_csv_file(file_path)
            except Exception:
                df_temp['manual_conf'] = 0.0

        if 'var_norm' not in df_temp.columns:
            df_temp['var_norm'] = 0.0

        df_temp['manual_conf'] = pd.to_numeric(df_temp['manual_conf'], errors='coerce').fillna(0.0)
        df_temp['var_norm'] = pd.to_numeric(df_temp['var_norm'], errors='coerce').fillna(0.0)

        if file not in classes_per_batch:
            classes_per_batch[file] = {}

        for i, label in enumerate(classes_list):
            if changed_slider_index is not None and i != changed_slider_index and (label in classes_per_batch[file]):
                continue

            conf_range = default_conf
            if slider_values_conf is not None and len(slider_values_conf) == len(classes_list) and slider_values_conf[i] is not None:
                conf_range = slider_values_conf[i]

            var_range = default_var
            if slider_values_var is not None and len(slider_values_var) == len(classes_list) and slider_values_var[i] is not None:
                var_range = slider_values_var[i]

            filtered = df_temp[
                (df_temp['manual_label'] == label) &
                (df_temp['manual_conf'] >= conf_range[0]) &
                (df_temp['manual_conf'] < conf_range[1]) &
                (df_temp['var_norm'] >= var_range[0]) &
                (df_temp['var_norm'] < var_range[1])
            ]
            classes_per_batch[file][label] = len(filtered)

        loadbar_classes = 20

    loadbar_classes = 100


def _make_range_folder_suffix(metric_name, rng):
    if rng is None:
        return ""
    a, b = rng
    a_i = int(round(a * 100))
    b_i = int(round(b * 100))
    if b >= 1.0:
        return f"{metric_name}>={a_i}"
    return f"{metric_name}>={a_i}and<{b_i}"


def save_dataset(output_folder, project_name, selected_classes, selected_batches,
                 slider_values_conf=None, slider_values_var=None,
                 clustering_enabled=False,
                 clustering_cuts_by_class=None,
                 apply_clustering_to_remaining=True):
    global loadbar_save_dataset

    count = 0
    project_folder = join(getcwd(), 'main', 'assets', project_name)
    batches_folder = join(project_folder, 'dataframes')
    images_folder = join(project_folder, 'images')

    if not exists(output_folder):
        mkdir(output_folder)

    default_conf = [0.0, 1.0]
    default_var = [0.0, 1.0]

    cluster_maps = {}
    cluster_cut_tags = {}

    if clustering_enabled and clustering_cuts_by_class and isinstance(clustering_cuts_by_class, dict):
        if fcluster is None:
            clustering_enabled = False
        else:
            for cls, cut01 in clustering_cuts_by_class.items():
                try:
                    cut01 = float(cut01)
                except Exception:
                    continue
                cut01 = min(max(cut01, 0.0), 1.0)

                artifact = _load_cluster_artifact(project_name, cls)
                if artifact is None:
                    continue

                names = artifact["names"]
                Z = artifact["linkage"]
                dist_max = float(artifact["dist_max"]) if float(artifact["dist_max"]) > 0 else 1.0

                cut_dist = cut01 * dist_max
                labels = fcluster(Z, t=cut_dist, criterion="distance")
                key = (project_name, cls, cut01)

                if key not in CLUSTER_ASSIGNMENT_CACHE:
                    CLUSTER_ASSIGNMENT_CACHE[key] = {str(n): int(c) for n, c in zip(names, labels)}

                cluster_maps[cls] = CLUSTER_ASSIGNMENT_CACHE[key]
                cluster_cut_tags[cls] = f"cut{int(round(100 * cut01))}"

    for label in selected_classes:
        class_index = selected_classes.index(label)

        conf_rng = default_conf
        if slider_values_conf is not None and class_index < len(slider_values_conf) and slider_values_conf[class_index] is not None:
            conf_rng = slider_values_conf[class_index]

        var_rng = default_var
        if slider_values_var is not None and class_index < len(slider_values_var) and slider_values_var[class_index] is not None:
            var_rng = slider_values_var[class_index]

        conf_suffix = _make_range_folder_suffix("conf", conf_rng)
        var_suffix = _make_range_folder_suffix("var", var_rng)

        extra = ""
        if clustering_enabled and (label in cluster_maps) and (label in cluster_cut_tags):
            extra = f"_{cluster_cut_tags[label]}"

        filtered_folder_name = f"{label}_{conf_suffix}_{var_suffix}{extra}".replace("__", "_").strip("_")
        filtered_label_path = join(output_folder, filtered_folder_name)
        if not exists(filtered_label_path):
            mkdir(filtered_label_path)

        remaining_folder_name = f"{label}_NOT_{conf_suffix}_{var_suffix}{extra}".replace("__", "_").strip("_")
        remaining_label_path = join(output_folder, remaining_folder_name)
        if not exists(remaining_label_path):
            mkdir(remaining_label_path)

    for batch in selected_batches:
        count += 1
        loadbar_save_dataset = 100 * count / max(1, len(selected_batches))

        batch_folder = batch[:-4]
        match = project_name + '.csv'
        if match in batch:
            match_size = len(match) + 1
            batch_folder = batch[:-match_size]

        temp_images_folder = join(images_folder, batch_folder)
        batch_images_folder = join(temp_images_folder, listdir(temp_images_folder)[0])

        csv_path = join(batches_folder, batch)
        df_temp = pd.read_csv(csv_path)
        df_temp = _ensure_manual_label(df_temp)

        if 'manual_conf' not in df_temp.columns:
            try:
                df_temp = update_csv_file(csv_path)
            except Exception:
                df_temp['manual_conf'] = 0.0

        if 'var_norm' not in df_temp.columns:
            df_temp['var_norm'] = 0.0

        df_temp['manual_conf'] = pd.to_numeric(df_temp['manual_conf'], errors='coerce').fillna(0.0)
        df_temp['var_norm'] = pd.to_numeric(df_temp['var_norm'], errors='coerce').fillna(0.0)

        df_filtered = df_temp.loc[df_temp['manual_label'].isin(selected_classes)]

        filtered_rows = []
        remaining_rows = []

        for i, label in enumerate(selected_classes):
            conf_rng = default_conf
            if slider_values_conf is not None and i < len(slider_values_conf) and slider_values_conf[i] is not None:
                conf_rng = slider_values_conf[i]

            var_rng = default_var
            if slider_values_var is not None and i < len(slider_values_var) and slider_values_var[i] is not None:
                var_rng = slider_values_var[i]

            class_filtered = df_filtered[
                (df_filtered['manual_label'] == label) &
                (df_filtered['manual_conf'] >= conf_rng[0]) &
                (df_filtered['manual_conf'] < conf_rng[1]) &
                (df_filtered['var_norm'] >= var_rng[0]) &
                (df_filtered['var_norm'] < var_rng[1])
            ]
            filtered_rows.append(class_filtered)

            class_remaining = df_filtered[
                (df_filtered['manual_label'] == label) &
                ~(
                    (df_filtered['manual_conf'] >= conf_rng[0]) &
                    (df_filtered['manual_conf'] < conf_rng[1]) &
                    (df_filtered['var_norm'] >= var_rng[0]) &
                    (df_filtered['var_norm'] < var_rng[1])
                )
            ]
            remaining_rows.append(class_remaining)

        def _copy_rows(df_rows, is_remaining: bool):
            if df_rows is None or df_rows.empty:
                return

            for _, row in df_rows.iterrows():
                name = row['names']
                label = row['manual_label']
                class_index = selected_classes.index(label)

                conf_rng = default_conf
                if slider_values_conf is not None and class_index < len(slider_values_conf) and slider_values_conf[class_index] is not None:
                    conf_rng = slider_values_conf[class_index]
                var_rng = default_var
                if slider_values_var is not None and class_index < len(slider_values_var) and slider_values_var[class_index] is not None:
                    var_rng = slider_values_var[class_index]

                conf_suffix = _make_range_folder_suffix("conf", conf_rng)
                var_suffix = _make_range_folder_suffix("var", var_rng)

                extra = ""
                if clustering_enabled and (label in cluster_maps) and (label in cluster_cut_tags):
                    extra = f"_{cluster_cut_tags[label]}"

                if not is_remaining:
                    folder_name = f"{label}_{conf_suffix}_{var_suffix}{extra}".replace("__", "_").strip("_")
                else:
                    folder_name = f"{label}_NOT_{conf_suffix}_{var_suffix}{extra}".replace("__", "_").strip("_")

                base_path = join(output_folder, folder_name)

                if clustering_enabled and (label in cluster_maps) and (not is_remaining or apply_clustering_to_remaining):
                    cid = cluster_maps[label].get(name, None)
                    sub = f"{label}_{int(cid)}" if cid is not None else f"{label}_UNK"
                    dest = join(base_path, sub)
                    if not exists(dest):
                        mkdir(dest)
                else:
                    dest = base_path

                copy2(join(batch_images_folder, name), join(dest, name))

        if filtered_rows:
            df_filtered_combined = pd.concat(filtered_rows, ignore_index=True)
            _copy_rows(df_filtered_combined, is_remaining=False)

        if remaining_rows:
            df_remaining_combined = pd.concat(remaining_rows, ignore_index=True)
            _copy_rows(df_remaining_combined, is_remaining=True)

    loadbar_save_dataset = 0
    return 0


def get_datasets_list():
    default_output_folder = join(getcwd(), 'output')
    if not exists(default_output_folder):
        makedirs(default_output_folder, exist_ok=True)
    datasets_list = [f for f in listdir(default_output_folder) if isdir(join(default_output_folder, f))]
    return datasets_list


# ======================================================================================
# Dash init
# ======================================================================================

get_projects_list()
datasets_list = get_datasets_list()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}]
)

DEFAULT_CLUSTER_CUT = 0.50

app.layout = html.Div([
    html.H1('ILT Data Explorer'),

    dcc.ConfirmDialog(id='confirm_save_dataset'),

    html.Datalist(
        id='list_suggested_datasets',
        children=[html.Option(value=name) for name in datasets_list]
    ),

    dbc.Container(
        dbc.Row([
            dbc.Col(dcc.Dropdown(projects_list, '', id='dropdown_project', clearable=False), width={"size": 12}),
        ]),
    ),

    # ALERTA para diagnosticar "sumiu tudo"
    dbc.Container(
        dbc.Row([
            dbc.Col(
                dbc.Alert(
                    id="alert_project_problem",
                    children="",
                    is_open=False,
                    color="warning",
                    dismissable=True,
                    duration=12000,
                    style={"marginTop": "8px"}
                ),
                width=12
            )
        ])
    ),

    dbc.Row([dbc.Col(html.Hr()), ], ),

    dbc.Container(
        dbc.Row([
            html.Div([
                dbc.Col(
                    [
                        html.Div([
                            dcc.Checklist(
                                options=[],
                                value=[],
                                id='checklist_batches',
                                labelStyle={'display': 'block'},
                                style={
                                    "overflow": "auto",
                                    'border': '1px solid #ddd',
                                    'borderRadius': '5px',
                                    'padding': '10px',
                                    'backgroundColor': '#f8f9fa',
                                    'height': '100%',
                                    'flexGrow': 1,
                                    'minHeight': 0
                                }
                            ),
                            html.Div([
                                dbc.Button('Update Classes & Histograms', n_clicks=0, id='button_update_classes',
                                           style={'background': 'steelblue', 'width': '100%'}),
                                dbc.Progress(value=0, id='progress_classes', style={'marginTop': '10px'})
                            ], style={'marginTop': '10px'})
                        ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'})
                    ], width=6
                ),
                dbc.Col([
                    html.Div([
                        html.Div(
                            id='slider-container',
                            style={
                                'height': '100%',
                                'overflow': 'auto',
                                'border': '1px solid #ddd',
                                'borderRadius': '5px',
                                'padding': '10px',
                                'backgroundColor': '#f8f9fa',
                                'flexGrow': 1,
                                'minHeight': 0
                            }
                        ),
                        html.Div([
                            dbc.Button('Update count', id='button_update_count', n_clicks=0,
                                       style={'background': 'seagreen', 'width': '100%'})
                        ], style={'marginTop': '10px'})
                    ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'})
                ], width=6
                ),
            ], style={'display': 'flex', 'height': '600px', 'gap': '15px', 'width': '100%'})
        ]),
    ),

    dbc.Row([dbc.Col(html.Hr()), ], ),

    # =======================
    # Clustering UI (CHECKLIST)
    # =======================
    dbc.Container(
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Class clustering (dendrogram / grupos)", style={"marginBottom": "10px"}),

                        dbc.Row([
                            dbc.Col(
                                dcc.Checklist(
                                    options=[{'label': ' Enable clustering for export', 'value': 'enable'}],
                                    value=[],
                                    id='check_enable_clustering'
                                ),
                                width=3
                            ),

                            dbc.Col(
                                html.Div([
                                    html.Div("Classes (marque 1 para visualizar):", style={"fontSize": "12px", "color": "#555"}),
                                    dcc.Checklist(
                                        id='checklist_cluster_classes',
                                        options=[],
                                        value=[],
                                        labelStyle={'display': 'block'},
                                        style={
                                            "overflow": "auto",
                                            "border": "1px solid #ddd",
                                            "borderRadius": "5px",
                                            "padding": "8px",
                                            "backgroundColor": "#ffffff",
                                            "height": "140px"
                                        }
                                    ),
                                    html.Div("Dica: defina cuts para várias classes, uma por vez (selecione 1 e clique Update).",
                                             style={"fontSize": "11px", "color": "#777", "marginTop": "6px"}),
                                ]),
                                width=3
                            ),

                            dbc.Col(
                                html.Div([
                                    html.Div("Cut (0..1 of max distance)", style={"fontSize": "12px", "color": "#555"}),
                                    dcc.Slider(
                                        min=0.0, max=1.0, step=0.01,
                                        value=DEFAULT_CLUSTER_CUT,
                                        id='slider_cluster_cut',
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ]),
                                width=4
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Update clusters (save cut for this class)",
                                    id='button_update_cluster',
                                    n_clicks=0,
                                    style={'background': 'indigo', 'width': '100%'}
                                ),
                                width=2
                            ),
                        ], align="center"),

                        dbc.Row([
                            dbc.Col(
                                dcc.Checklist(
                                    options=[{'label': ' Apply clustering to remaining (NOT) export', 'value': 'enable'}],
                                    value=['enable'],
                                    id='check_cluster_apply_remaining'
                                ),
                                width=12
                            ),
                        ], style={"marginTop": "6px"}),

                        html.Div(
                            id='cluster_summary',
                            style={"marginTop": "10px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"}
                        ),

                        dcc.Graph(
                            id='graph_cluster_dendrogram',
                            figure=go.Figure(),
                            config={'displayModeBar': False},
                            style={"marginTop": "6px"}
                        ),

                        dbc.Alert(
                            id="alert_cluster_problem",
                            children="",
                            dismissable=True,
                            is_open=False,
                            color="warning",
                            duration=6000
                        ),
                    ]),
                    style={"backgroundColor": "#f8f9fa"}
                ),
                width=12
            )
        ])
    ),

    dbc.Row([dbc.Col(html.Hr()), ], ),

    dbc.Container(
        dbc.Row([
            dbc.Col(
                dbc.Input(
                    value='dataset',
                    id='input_save_dataset',
                    type="text",
                    style={'width': '100%', 'background': 'Floralwhite'},
                    list='list_suggested_datasets'
                ),
            ),
            dbc.Col([
                dbc.Button('Save Dataset', n_clicks=0, id='button_save_dataset',
                           style={'background': 'chocolate', 'width': '100%'}),
                dbc.Progress(value=0, id="progress_save_dataset"),
                dbc.Alert(children='', id="alert_save_dataset_problem", dismissable=True, is_open=False, color='warning',
                          duration=5000),
                dbc.Alert(children='', id="alert_save_dataset_success", dismissable=True, is_open=False),
            ])
        ])
    ),

    dcc.Interval(id='clock', interval=1000, n_intervals=0, max_intervals=-1),
    dcc.Store(id='button_display_classes_nclicks', data=0),
    dcc.Store(id='button_update_classes_nclicks', data=0),
    dcc.Store(id='dataset_name_ok', data=False),
    dcc.Store(id='save_dataset_result', data=0),

    # classes do projeto (p/ checklist de cluster sem duplicar outputs)
    dcc.Store(id='classes_store', data=json.dumps({"project": "", "classes": []})),

    # meta leve: cuts por classe
    dcc.Store(id='cluster_meta_store', data=json.dumps({"ready": False, "project": "", "cuts": {}, "last_class": "", "last_cut": DEFAULT_CLUSTER_CUT})),
])


# ======================================================================================
# Progress callbacks
# ======================================================================================

@app.callback(
    Output("progress_save_dataset", "value"),
    Input("clock", "n_intervals"))
def progress_save_dataset_update(n):
    global loadbar_save_dataset
    return loadbar_save_dataset


@app.callback(
    Output("progress_classes", "value"),
    Input("clock", "n_intervals"))
def progress_classes_update_separate(n):
    global loadbar_classes
    return loadbar_classes


# ======================================================================================
# Update batches list on project change (COM DIAGNÓSTICO)
# ======================================================================================

@app.callback(
    Output('checklist_batches', 'options'),
    Output('checklist_batches', 'value'),
    Output('input_save_dataset', 'value'),
    Output('alert_project_problem', 'children'),
    Output('alert_project_problem', 'is_open'),
    Input('dropdown_project', 'value')
)
def update_batches_list(project_name):
    global classes_per_batch, loadbar_classes, loadbar_save_dataset, previous_slider_values, previous_var_slider_values

    if not project_name:
        return [], [], 'dataset', '', False

    try:
        batches_folder = join(getcwd(), 'main', 'assets', project_name, 'dataframes')
        if not exists(batches_folder):
            return [], [], project_name + '_dataset', f"Pasta não encontrada: {batches_folder}", True

        batches_list = [f for f in listdir(batches_folder) if f.lower().endswith('.csv')]
        batches_list.sort()

        if not batches_list:
            return [], [], project_name + '_dataset', f"Nenhum .csv encontrado em: {batches_folder}", True

        batches_text = [{'label': ' ' + str(i + 1) + ': ' + batches_list[i], 'value': batches_list[i]}
                        for i in range(len(batches_list))]

        classes_per_batch = {}
        loadbar_classes = 0
        loadbar_save_dataset = 0
        previous_slider_values = []
        previous_var_slider_values = []

        return batches_text, batches_list, project_name + '_dataset', '', False

    except Exception as e:
        return [], [], project_name + '_dataset', f"Erro ao listar batches: {type(e).__name__}: {e}", True


# ======================================================================================
# Update classes/histograms (e guarda classes_store)
# ======================================================================================

@app.callback(
    Output('slider-container', 'children'),
    Output('button_display_classes_nclicks', 'data'),
    Output('button_update_classes_nclicks', 'data'),
    Output('classes_store', 'data'),
    Input('button_update_classes', 'n_clicks'),
    Input('button_update_count', 'n_clicks'),
    State('slider-container', 'children'),
    State({'type': 'dynamic-slider', 'index': dash.ALL}, 'value'),          # manual_conf sliders
    State({'type': 'dynamic-slider-var', 'index': dash.ALL}, 'value'),      # var_norm sliders
    State('dropdown_project', 'value'),
    State('checklist_batches', 'value'),
    State('button_display_classes_nclicks', 'data'),
    State('button_update_classes_nclicks', 'data')
)
def update_classes_list(button_update_classes_clicks, button_update_count_clicks, slide_container,
                        slider_values_conf, slider_values_var,
                        project_name, checklist_value,
                        prev_nclicks, prev_button_clicks):
    global current_classes_list, previous_slider_values, previous_var_slider_values, loadbar_classes
    components = []

    ctx = dash.callback_context
    button_triggered = False
    trigger_id = None

    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'button_update_classes' and button_update_classes_clicks and button_update_classes_clicks > prev_button_clicks:
            button_triggered = True
        elif trigger_id == 'button_update_count' and button_update_count_clicks and button_update_count_clicks > 0:
            button_triggered = True

    if not button_triggered:
        return components, prev_nclicks, prev_button_clicks, dash.no_update

    if not project_name or not checklist_value:
        return components, prev_nclicks, prev_button_clicks, json.dumps({"project": project_name or "", "classes": []})

    batches_folder = join(getcwd(), 'main', 'assets', project_name, 'dataframes')

    # tenta corrigir manual_conf quando faltar
    for batch_file in checklist_value:
        file_path = join(batches_folder, batch_file)
        try:
            df = pd.read_csv(file_path, nrows=50)
            df = _ensure_manual_label(df)
            if 'manual_conf' not in df.columns:
                _ = update_csv_file(file_path)
        except Exception:
            pass

    classes_list = get_classes(batches_folder, checklist_value)

    get_classes_per_batch(
        batches_folder, checklist_value, classes_list,
        slider_values_conf=slider_values_conf,
        slider_values_var=slider_values_var,
        changed_slider_index=None
    )

    classes_text, classes_list = update_classes_project(checklist_value, project_name)

    hist_data_conf = get_histogram_data_metric(batches_folder, checklist_value, classes_list, metric_col='manual_conf')
    hist_data_var = get_histogram_data_metric(batches_folder, checklist_value, classes_list, metric_col='var_norm')

    current_classes_list = classes_list

    if slider_values_conf:
        previous_slider_values = slider_values_conf.copy()
    if slider_values_var:
        previous_var_slider_values = slider_values_var.copy()

    for i in range(len(classes_text)):
        slider_conf_value = slider_values_conf[i] if slider_values_conf and i < len(slider_values_conf) and slider_values_conf[i] is not None else [0.0, 1.0]
        slider_var_value = slider_values_var[i] if slider_values_var and i < len(slider_values_var) and slider_values_var[i] is not None else [0.0, 1.0]

        hist_fig_conf = go.Figure()
        hist_fig_conf.add_trace(go.Histogram(
            x=hist_data_conf.get(classes_list[i], []),
            xbins=dict(start=0, end=1.001, size=0.05),
            opacity=0.7,
            nbinsx=20
        ))
        hist_fig_conf.add_vline(x=slider_conf_value[0], line_dash="dash", line_color="red")
        hist_fig_conf.add_vline(x=slider_conf_value[1], line_dash="dash", line_color="red")
        hist_fig_conf.update_layout(
            height=48, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
            xaxis=dict(showticklabels=False, range=[-0.07, 1.07], showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
        )
        histogram_conf = dcc.Graph(
            id={'type': 'dynamic-histogram', 'index': i},
            figure=hist_fig_conf,
            config={'displayModeBar': False}
        )
        slider_conf = dcc.RangeSlider(
            id={'type': 'dynamic-slider', 'index': i},
            min=0, max=1.0, step=0.05,
            value=slider_conf_value,
            marks={0: '0', 1: '1'},
            tooltip={"placement": "bottom", "always_visible": True,
                     "style": {"color": "DarkBlue", "fontSize": "16px"}}
        )

        hist_fig_var = go.Figure()
        hist_fig_var.add_trace(go.Histogram(
            x=hist_data_var.get(classes_list[i], []),
            xbins=dict(start=0, end=1.001, size=0.05),
            opacity=0.7,
            nbinsx=20
        ))
        hist_fig_var.add_vline(x=slider_var_value[0], line_dash="dash", line_color="red")
        hist_fig_var.add_vline(x=slider_var_value[1], line_dash="dash", line_color="red")
        hist_fig_var.update_layout(
            height=48, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
            xaxis=dict(showticklabels=False, range=[-0.07, 1.07], showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
        )
        histogram_var = dcc.Graph(
            id={'type': 'dynamic-histogram-var', 'index': i},
            figure=hist_fig_var,
            config={'displayModeBar': False}
        )
        slider_var = dcc.RangeSlider(
            id={'type': 'dynamic-slider-var', 'index': i},
            min=0, max=1.0, step=0.05,
            value=slider_var_value,
            marks={0: '0', 1: '1'},
            tooltip={"placement": "bottom", "always_visible": True,
                     "style": {"color": "DarkBlue", "fontSize": "16px"}}
        )

        checklist = dcc.Checklist(
            id={'type': 'dynamic-checklist', 'index': i},
            options=[classes_text[i]],
            value=['enable'],
            style={'marginLeft': '0px', 'width': '25%'}
        )

        conf_block = html.Div([
            html.Div("manual_conf", style={"fontSize": "12px", "color": "#555"}),
            histogram_conf,
            html.Div(slider_conf, style={'marginTop': '-16px'})
        ], style={'width': '35%'})

        var_block = html.Div([
            html.Div("var_norm", style={"fontSize": "12px", "color": "#555"}),
            histogram_var,
            html.Div(slider_var, style={'marginTop': '-16px'})
        ], style={'width': '35%'})

        row = html.Div(
            [checklist, conf_block, var_block],
            style={'display': 'flex', 'alignItems': 'flex-start', 'gap': '15px', 'marginBottom': '15px'}
        )

        loadbar_classes = 80 + 20 * i / max(1, len(classes_text))
        components.append(row)

    loadbar_classes = 100

    store_payload = json.dumps({"project": project_name, "classes": classes_list})

    if trigger_id == 'button_update_classes':
        return components, prev_nclicks, button_update_classes_clicks, store_payload
    else:
        return components, prev_nclicks + 1, prev_button_clicks, store_payload


# ======================================================================================
# Cluster checklist options (sem duplicar outputs)
# ======================================================================================

@app.callback(
    Output('checklist_cluster_classes', 'options'),
    Output('checklist_cluster_classes', 'value'),
    Input('classes_store', 'data'),
    Input('cluster_meta_store', 'data'),
    State('dropdown_project', 'value'),
    State('checklist_cluster_classes', 'value'),
)
def refresh_cluster_checklist(classes_store_json, cluster_meta_json, project_name, selected_now):
    try:
        payload = json.loads(classes_store_json) if classes_store_json else {}
    except Exception:
        payload = {}

    classes_list = []
    if payload.get("project") == project_name and isinstance(payload.get("classes"), list):
        classes_list = [str(x) for x in payload["classes"]]

    opts = _cluster_checklist_options(project_name, classes_list) if project_name else []
    selected_now = selected_now or []
    selected_kept = [v for v in selected_now if v in classes_list]
    return opts, selected_kept


# ======================================================================================
# Clustering update callback (cut POR CLASSE)
# ======================================================================================

@app.callback(
    Output('cluster_meta_store', 'data'),
    Output('graph_cluster_dendrogram', 'figure'),
    Output('cluster_summary', 'children'),
    Output('alert_cluster_problem', 'children'),
    Output('alert_cluster_problem', 'is_open'),
    Input('button_update_cluster', 'n_clicks'),
    State('dropdown_project', 'value'),
    State('checklist_cluster_classes', 'value'),
    State('slider_cluster_cut', 'value'),
)
def update_cluster_view(nclicks, project_name, selected_classes_for_cluster, cut01):
    if not nclicks:
        meta0 = {"ready": False, "project": "", "cuts": {}, "last_class": "", "last_cut": DEFAULT_CLUSTER_CUT}
        return json.dumps(meta0), go.Figure(), "", "", False

    selected = selected_classes_for_cluster or []
    if not project_name or not selected:
        meta0 = {"ready": False, "project": project_name or "", "cuts": _cuts_for_project(project_name or ""), "last_class": "", "last_cut": float(cut01 or DEFAULT_CLUSTER_CUT)}
        return json.dumps(meta0), go.Figure(), "", "Select project and 1 class first.", True

    if len(selected) != 1:
        meta0 = {"ready": False, "project": project_name, "cuts": _cuts_for_project(project_name), "last_class": "", "last_cut": float(cut01 or DEFAULT_CLUSTER_CUT)}
        return json.dumps(meta0), go.Figure(), "", "Marque APENAS 1 classe para visualizar o dendrograma.", True

    class_label = selected[0]

    if fcluster is None:
        meta0 = {"ready": False, "project": project_name, "cuts": _cuts_for_project(project_name), "last_class": class_label, "last_cut": float(cut01)}
        return json.dumps(meta0), go.Figure(), "", "scipy não está disponível. Instale scipy para clustering/dendrograma.", True

    artifact = _load_cluster_artifact(project_name, class_label)
    if artifact is None:
        msg = f"Cluster artifact não encontrado para '{class_label}'. Gere com class_clustering.py (saída em main/assets/<project>/clusters/<class>.npz)"
        meta0 = {"ready": False, "project": project_name, "cuts": _cuts_for_project(project_name), "last_class": class_label, "last_cut": float(cut01)}
        return json.dumps(meta0), go.Figure(), "", msg, True

    names = artifact["names"]
    Z = artifact["linkage"]
    dist_max = float(artifact["dist_max"]) if float(artifact["dist_max"]) > 0 else 1.0

    cut01 = float(cut01)
    cut01 = min(max(cut01, 0.0), 1.0)
    cut_dist = cut01 * dist_max

    labels = fcluster(Z, t=cut_dist, criterion="distance")
    K = int(labels.max()) if labels.size else 0

    # salva o cut ANTES de tentar desenhar o dendrograma
    CLUSTER_CUTS[(project_name, class_label)] = cut01

    key = (project_name, class_label, cut01)
    if key not in CLUSTER_ASSIGNMENT_CACHE:
        CLUSTER_ASSIGNMENT_CACHE[key] = {str(n): int(c) for n, c in zip(names, labels)}

    counts = Counter(labels.tolist())
    largest = max(counts.values()) if counts else 0

    summary_main = (
    f"[last] class='{class_label}' | N={len(names)} | cut={cut01:.2f} "
    f"(dist={cut_dist:.6g}/{dist_max:.6g}) | groups={K} | largest={largest}\n\n"
    f"Cuts por classe aparecem ao lado do nome na caixa de seleção."
    )

    # dendrogram protegido: se falhar, cut permanece salvo e o callback não quebra
    alert_msg = ""
    alert_open = False
    try:
        fig = _build_dendrogram_figure(Z, cut_dist=cut_dist, max_leaves=1500)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Dendrograma não pôde ser desenhado: {type(e).__name__}: {e}\nCut foi salvo normalmente.",
            showarrow=False
        )
        alert_msg = f"Dendrograma falhou (cut salvo): {type(e).__name__}: {e}"
        alert_open = True

    meta = {
        "ready": True,
        "project": project_name,
        "cuts": _cuts_for_project(project_name),
        "last_class": class_label,
        "last_cut": cut01,
    }
    return json.dumps(meta), fig, summary_main, alert_msg, alert_open


# ======================================================================================
# Save dataset flow (clustering per-class cuts)
# ======================================================================================

@app.callback(
    Output('alert_save_dataset_success', 'children'),
    Output('alert_save_dataset_success', 'is_open'),
    Input('confirm_save_dataset', 'submit_n_clicks'),
    State('dropdown_project', 'value'),
    State('input_save_dataset', 'value'),
    State('checklist_batches', 'value'),
    State({'type': 'dynamic-checklist', 'index': dash.ALL}, 'value'),
    State({'type': 'dynamic-slider', 'index': dash.ALL}, 'value'),       # manual_conf
    State({'type': 'dynamic-slider-var', 'index': dash.ALL}, 'value'),   # var_norm
    State('check_enable_clustering', 'value'),
    State('check_cluster_apply_remaining', 'value'),
    State('cluster_meta_store', 'data'),
)
def save_dataset_confirmed(nclicks, project_name, dataset_name, selected_batches, selected_classes,
                           slider_values_conf, slider_values_var,
                           clustering_enable_value,
                           apply_remaining_value,
                           cluster_meta_json):
    global current_classes_list

    if not nclicks:
        return '', False

    actual_selected_classes = []
    if selected_classes and current_classes_list:
        for i, class_list in enumerate(selected_classes):
            if class_list and i < len(current_classes_list):
                actual_selected_classes.append(current_classes_list[i])

    default_output_folder = join(getcwd(), 'output')
    output_folder = join(default_output_folder, dataset_name)

    clustering_enabled = (clustering_enable_value == ['enable'])
    apply_remaining = (apply_remaining_value == ['enable'])

    cuts_by_class = {}
    try:
        meta = json.loads(cluster_meta_json) if cluster_meta_json else {}
        if meta.get("project") == project_name and isinstance(meta.get("cuts"), dict):
            cuts_by_class = {str(k): float(v) for k, v in meta["cuts"].items()}
    except Exception:
        cuts_by_class = {}

    cuts_selected = {c: cuts_by_class[c] for c in actual_selected_classes if c in cuts_by_class}

    if clustering_enabled and not cuts_selected:
        clustering_enabled = False

    save_dataset(
        output_folder, project_name, actual_selected_classes, selected_batches,
        slider_values_conf, slider_values_var,
        clustering_enabled=clustering_enabled,
        clustering_cuts_by_class=cuts_selected,
        apply_clustering_to_remaining=apply_remaining
    )
    return dataset_name + ' saved successfully', True


@app.callback(
    Output('confirm_save_dataset', 'message'),
    Output('confirm_save_dataset', 'displayed'),
    Input('dataset_name_ok', 'data'),
    State('input_save_dataset', 'value'),
)
def save_dataset_confirmation(name_ok, dataset_name):
    default_output_folder = join(getcwd(), 'output')
    output_folder = join(default_output_folder, dataset_name)

    if name_ok is True:
        if exists(output_folder):
            return dataset_name + ' already exists. Do you really want to merge these images into the existing dataset?', True
        else:
            return dataset_name + ' will be created. Proceed?', True
    return '', False


@app.callback(
    Output('alert_save_dataset_problem', 'children'),
    Output('alert_save_dataset_problem', 'is_open'),
    Output('dataset_name_ok', 'data'),
    Input('button_save_dataset', 'n_clicks'),
    State('dropdown_project', 'value'),
    State('checklist_batches', 'value'),
    State({'type': 'dynamic-checklist', 'index': dash.ALL}, 'value'),
)
def click_save_dataset(nclicks, project_name, selected_batches, selected_classes):
    global current_classes_list
    if nclicks > 0:
        default_output_folder = join(getcwd(), 'output')
        if not exists(default_output_folder):
            mkdir(default_output_folder)

        actual_selected_classes = []
        if selected_classes and current_classes_list:
            for i, class_list in enumerate(selected_classes):
                if class_list and i < len(current_classes_list):
                    actual_selected_classes.append(current_classes_list[i])

        if project_name == '':
            return 'No project was selected', True, False
        elif len(selected_batches) == 0:
            return 'No batches were selected', True, False
        elif len(actual_selected_classes) == 0:
            return 'No classes were selected', True, False
        else:
            return '', False, True
    return '', False, False


# ======================================================================================
# Run
# ======================================================================================

port = 8030
opened = False

if not opened:
    webbrowser.open('http://127.0.0.1:' + str(port) + '/', new=2, autoraise=True)
    opened = True

if __name__ == '__main__':
    app.title = 'ILT Data Explorer'
    app.run_server(debug=False, port=port)

