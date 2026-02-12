import sys

# Você usa argv[3] (csv), então precisa de pelo menos 4 args:
# app.py <path_to_images> <path_to_thumbnails> <csv_file> [graph_range]
assert len(sys.argv) >= 4, "Uso: python app.py <imgs_path> <thumbs_path> <csv_path> [graph_range]"

from functions import *

import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import imageselector
import json
import ast

from os import mkdir, listdir
from os.path import exists, join, dirname, basename, isdir, splitext
import shutil
import webbrowser
from datetime import date
from pandas.api.types import is_numeric_dtype


def _infer_project_name_from_images_path(path_to_images: str) -> str:
    """
    Tenta inferir o nome do projeto a partir de um caminho como:
      assets/<project>/images/...
      main/assets/<project>/images/...
    Fallback: segundo componente do path.
    """
    parts = [p for p in path_to_images.replace("\\", "/").split("/") if p]
    if "assets" in parts:
        i = parts.index("assets")
        if i + 1 < len(parts):
            return parts[i + 1]
    if len(parts) >= 2:
        return parts[1]
    return parts[0] if parts else "project"


def init(argv):
    """
    argv:
      [1] path to images (ex: assets/<project>/images/batch_0001/samples/)
      [2] path to thumbnails (ex: assets/<project>/thumbnails/batch_0001/samples/)
      [3] input CSV (ex: main/assets/<project>/dataframes/batch_0001_<project>.csv)
      [4] graph_range (opcional)
    """
    path_to_images = argv[1]
    path_to_thumbnails = argv[1]
    csv_file = argv[3]

    graph_range = 80
    if len(argv) > 4:
        graph_range = int(argv[4])

    print("imgs:", path_to_images)
    print("thumbnails:", path_to_thumbnails)
    print("csv:", csv_file)

    project_name = _infer_project_name_from_images_path(path_to_images)

    csv_folder = dirname(csv_file)
    csv_basename = basename(csv_file)
    batch_str = csv_basename[:-4]

    # Background opcional
    background_base = join("main", "assets", project_name, "backgrounds")
    background_path = None

    if isdir(background_base):
        matches = [f for f in listdir(background_base) if splitext(f)[0] == batch_str]
        if matches:
            background_path = join(background_base, matches[0])
        else:
            print(f"[WARN] No background file for '{batch_str}' in {background_base}. Running without background.")
    else:
        print(f"[WARN] Backgrounds folder not found: {background_base}. Running without background.")

    user_id = 0
    port = 8025

    return (
        path_to_images,
        path_to_thumbnails,
        csv_file,
        csv_folder,
        csv_basename,
        background_path,
        user_id,
        port,
        graph_range
    )


def read_input_csv(csv_file):
    return pd.read_csv(csv_file, encoding="ISO-8859-1")


def init_appearance():
    return "rgba(255, 250, 240, 100)"


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante colunas mínimas pra não quebrar:
      - names (obrigatório)
      - manual_label / correct_label
      - colors
      - custom_data (id único)
    """
    if "names" not in df.columns:
        raise ValueError("CSV não possui coluna 'names' (obrigatória).")

    # labels
    if "manual_label" not in df.columns:
        if "correct_label" in df.columns:
            df["manual_label"] = df["correct_label"].astype(str)
        else:
            df["manual_label"] = "unlabeled"

    if "correct_label" not in df.columns:
        df["correct_label"] = df["manual_label"].astype(str)

    # colors
    if "colors" not in df.columns:
        # cria ids por label (estável)
        uniq = sorted(df["manual_label"].astype(str).unique().tolist())
        lab2id = {lab: i for i, lab in enumerate(uniq)}
        df["colors"] = df["manual_label"].astype(str).map(lab2id).astype(int)
    else:
        df["colors"] = pd.to_numeric(df["colors"], errors="coerce").fillna(0).astype(int)

    # custom_data: id único por linha (usado no scatter e seleção)
    if "custom_data" not in df.columns:
        df["custom_data"] = df["names"].astype(str)
        if df["custom_data"].duplicated().any():
            df["custom_data"] = [f"{n}__{i}" for i, n in enumerate(df["names"].astype(str))]
    else:
        df["custom_data"] = df["custom_data"].astype(str)
        if df["custom_data"].duplicated().any():
            # força unicidade
            df["custom_data"] = [f"{cd}__{i}" for i, cd in enumerate(df["custom_data"].astype(str))]

    return df


# =========================
# INIT
# =========================
(
    path_to_images,
    path_to_thumbnails,
    csv_file,
    csv_folder,
    csv_basename,
    background_path,
    user_id,
    port,
    graph_range
) = init(sys.argv)

df = read_input_csv(csv_file)
df = _ensure_required_columns(df)

thumbs_jpg = ("thumbnails" in df.columns)  # se CSV traz thumbs por linha
HAS_VAR = ("var_norm" in df.columns)

# ratios
widths, heights = compute_img_ratios(path_to_images, df["names"])
df["widths"] = widths
df["heights"] = heights

background_ranges = {"x": [-graph_range, graph_range], "y": [-graph_range, graph_range]}
init_par_coords = False

background_color = init_appearance()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
)

labels_buffer = None
colors_buffer = None

# mapeamento label -> id cor (estável)
labels_colors = {}
for lab in df["manual_label"].astype(str).unique().tolist():
    # pega o primeiro id que aparece para esse label
    labels_colors[lab] = int(df.loc[df["manual_label"].astype(str) == lab, "colors"].iloc[0])

next_label_id = int(df["colors"].max()) + 1

# =========================
# UI COMPONENTS
# =========================
header = html.H1("Image Labeling Tool - " + csv_basename[:-4], style={"color": "CornflowerBlue"})

button_group_2 = dbc.ButtonGroup(
    [
        dbc.Input(
            placeholder="enter label here",
            id="input_aplicar_novo_label",
            type="text",
            style={"width": "50%", "background": "Floralwhite"},
        ),
        dbc.Button(
            "Label",
            n_clicks=0,
            id="button_aplicar_novo_label",
            style={"background": "chocolate", "width": "50%"},
        ),
    ],
    className="my-btn-group",
    size="lg",
)

button_group_3 = dbc.ButtonGroup(
    [
        dbc.Input(value=csv_basename, id="input_save_csv", type="text", style={"width": "50%", "background": "Floralwhite"}),
        dbc.Button("Save CSV", n_clicks=0, id="button_save_csv", style={"background": "chocolate", "width": "50%"}),
    ],
    className="my-btn-group",
    size="lg",
)

button_group_4 = dbc.ButtonGroup(
    [
        dbc.Input(value="annotated_dataset", id="input_finish_work", type="text", style={"width": "50%", "background": "Floralwhite"}),
        dbc.Button("Save Annotated Dataset", n_clicks=0, id="button_finish_work", style={"background": "chocolate", "width": "50%"}),
    ],
    className="my-btn-group",
    size="lg",
)

button_group_5 = dbc.ButtonGroup(
    [dbc.Button("UNDO", n_clicks=0, id="button_undo", style={"background": "firebrick", "width": "100%"})],
    className="my-btn-group",
    size="lg",
)

button_group_6 = dbc.ButtonGroup(
    [dbc.Button("Invert marks", n_clicks=0, id="button_invert_marks", style={"background": "chocolate", "width": "100%"})],
    className="my-btn-group",
    size="lg",
)

button_group_7 = dbc.ButtonGroup(
    [dcc.Checklist([" Save CSV after labeling"], value=[" Save CSV after labeling"], id="check_save_csv")],
    className="my-btn-group",
    size="lg",
)

button_group_8 = dbc.ButtonGroup(
    [dcc.Checklist([" Discard images after labeling"], value=[" Discard images after labeling"], id="check_discard_marked")],
    className="my-btn-group",
    size="lg",
)

button_group_9 = dbc.ButtonGroup(
    [
        dcc.Checklist([" Marked images first"], value=[], id="check_marked_first"),
        dcc.Checklist([" Hide relabeled images"], value=[], id="check_discard_relabeled"),
    ],
    vertical=True,
    className="my-btn-group",
    size="lg",
)

IMAGES = create_list_dics(_list_src=[], _list_thumbnail=[], _list_name_figure=[])

# Scatter (background opcional)
fig = f_figure_scatter_plot(
    df,
    _columns=["x", "y"],
    _selected_custom_data=[],
    background_img=background_path if (background_path and exists(background_path)) else None,
    xrange=background_ranges["x"],
    yrange=background_ranges["y"],
)

# ParCoords robusto
PREFERRED_PARCOLS = [
    "x", "y",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7",
    "x2", "y2", "x3", "y3",
    "confs1", "confs2", "max_prob", "var_norm",
    "area", "perimeter", "circularity", "avg_contour", "avg_intensity", "elongation",
]
numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

_columns_paralelas_coordenadas = []
for c in PREFERRED_PARCOLS:
    if c in numeric_cols and c not in _columns_paralelas_coordenadas:
        _columns_paralelas_coordenadas.append(c)

for c in numeric_cols:
    if c not in _columns_paralelas_coordenadas and c not in ("colors",):
        _columns_paralelas_coordenadas.append(c)

_columns_paralelas_coordenadas = _columns_paralelas_coordenadas[:7]

if len(_columns_paralelas_coordenadas) == 0:
    fig_paral = go.Figure()
    fig_paral.update_layout(
        title="Parallel Coordinates (sem colunas numéricas disponíveis no CSV)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
else:
    fig_paral = f_figure_paralelas_coordenadas(
        _df=df,
        _filtered_df=pd.DataFrame(columns=df.columns),
        _columns=_columns_paralelas_coordenadas,
        _selected_custom_data=[],
        _fig=None,
    )

# Dropdowns / sort
dropdown_image_vals = ["A-Z, a-z", "Similarity"]
map_columns_vals = {
    "area": "Seg Area",
    "perimeter": "Perimeter",
    "circularity": "Circularity",
    "avg_contour": "Avg Contour",
    "avg_intensity": "Avg Intensity",
    "elongation": "Elongation",
}

if "confs1" in df.columns:
    dropdown_image_vals.append("Confidence")
    if "confs2" in df.columns:
        dropdown_image_vals.append("Conf diff")

if HAS_VAR:
    dropdown_image_vals.append("Variance")

for key in map_columns_vals.keys():
    if key in df.columns:
        dropdown_image_vals.append(map_columns_vals[key])

dropdown_image_vals.sort()

# =========================
# LAYOUT
# =========================
app.layout = html.Div(
    [
        dbc.Container(
            dbc.Row(
                [dbc.Col(html.Div(header))],
                style={"max-height": "128px", "color": "white"},
            ),
            className="d-flex align-content-end",
            style={"max-width": "100%", "background-color": "Floralwhite"},
        ),
        dbc.Row([dbc.Col(html.Hr())]),
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(html.Div(html.P("Opacity"), style={"textAlign": "left"}), width={"size": 2}),
                                    dbc.Col(
                                        [
                                            dcc.Slider(
                                                0, 1, 0.1,
                                                value=0,
                                                id="slider_map_opacity",
                                                marks=None,
                                                tooltip={"placement": "bottom", "always_visible": True},
                                            )
                                        ],
                                        width={"size": 3},
                                    ),
                                    dbc.Col(html.Div(html.P("Marker"), style={"textAlign": "right"}), width={"size": 2}),
                                    dbc.Col(
                                        [
                                            dcc.Slider(
                                                1, 25, 3,
                                                value=10,
                                                id="slider_marker_size",
                                                marks=None,
                                                tooltip={"placement": "bottom", "always_visible": True},
                                            )
                                        ],
                                        width={"size": 3},
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            ["A-Z, a-z", "Frequency"],
                                            value="A-Z, a-z",
                                            id="dropdown_order_labels",
                                            clearable=False,
                                        ),
                                        width={"size": 2},
                                    ),
                                ],
                                align="bottom",
                            ),
                            dcc.Graph(
                                id="g_scatter_plot",
                                figure=fig,
                                style={"height": "74vh"},
                                config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage", "resetScale2d"]},
                            ),
                        ],
                        width={"size": 7},
                    ),
                    dbc.Col(
                        [
                            dbc.Row([dbc.Col(button_group_7, width={"size": 6}), dbc.Col(button_group_8, width={"size": 6})]),
                            dbc.Row([dbc.Col(button_group_2, width={"size": 12})]),
                            dbc.Row([dbc.Col(html.Hr())]),
                            dbc.Row(
                                [
                                    dbc.Col(button_group_6, width={"size": 3}),
                                    dbc.Col(button_group_9, width={"size": 4}),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            dropdown_image_vals,
                                            value=["Similarity"],
                                            id="dropdown_order_images",
                                            clearable=False,
                                            multi=True,
                                        ),
                                        width={"size": 5},
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Row(html.Div(html.P("Confidence range"), style={"textAlign": "center"})),
                                            dbc.Row(
                                                dcc.RangeSlider(
                                                    0, 1, 0.05,
                                                    value=[0, 1],
                                                    id="probs_range_slider",
                                                    marks=None,
                                                    tooltip={"placement": "bottom", "always_visible": True},
                                                )
                                            ),
                                        ],
                                        width={"size": 6},
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Row(html.Div(html.P("Variance range"), style={"textAlign": "center"})),
                                            dbc.Row(
                                                dcc.RangeSlider(
                                                    0, 1, 0.05,
                                                    value=[0, 1],
                                                    id="vars_range_slider",
                                                    marks=None,
                                                    tooltip={"placement": "bottom", "always_visible": True},
                                                )
                                            ),
                                        ],
                                        width={"size": 6},
                                        style={"display": "block" if HAS_VAR else "none"},
                                    ),
                                ]
                            ),
                            dbc.Row(
                                html.Div(
                                    imageselector.ImageSelector(
                                        id="g_image_selector",
                                        images=IMAGES,
                                        galleryHeaderStyle={
                                            "position": "sticky",
                                            "top": 0,
                                            "height": "0px",
                                            "background": "#000000",
                                            "zIndex": -1,
                                        },
                                    ),
                                    id="XXXXXXXXXX",
                                    style=dict(height="63vh", overflow="scroll", backgroundColor=background_color),
                                )
                            ),
                        ],
                        width={"size": 5},
                    ),
                ]
            ),
            style={"max-width": "100%"},
        ),
        dbc.Row([dbc.Col(html.Hr())]),
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            id="g_coordenadas_paralelas",
                            figure=fig_paral,
                            style={"height": "70vh"},
                            config={"displayModeBar": False},
                        ),
                        style=dict(width="100%", overflow="scroll"),
                        width={"size": 12},
                    )
                ]
            ),
            style={"max-width": "100%"},
        ),
        dbc.Row([dbc.Col(html.Hr())]),
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(button_group_3, width={"size": 4}),
                    dbc.Col(width={"size": 1}),
                    dbc.Col(button_group_4, width={"size": 4}),
                    dbc.Col(width={"size": 1}),
                    dbc.Col(button_group_5, width={"size": 2}),
                ]
            ),
            style={"max-width": "100%"},
        ),
        dcc.Store(id="selected_custom_points", data=json.dumps([])),
        dcc.Store(id="unchecked_points", data=json.dumps([])),
        dcc.Store(id="chart_flag", data=0),
        dcc.Store(id="state_store_df", data=df.to_json()),
        dcc.Store(id="dummy_csv_save", data=0),
        dcc.Store(id="dummy_finish_work", data=0),
        dcc.Store(id="dummy_map_images", data=0),
        html.Div(id="output"),
    ]
)

# =========================
# HELPERS
# =========================
def save_csv(df_updated, csv_name):
    global csv_folder
    filename = join(csv_folder, csv_name)
    df_updated.to_csv(filename, index=False)

    backup_path = join(csv_folder, "backups")
    if not exists(backup_path):
        mkdir(backup_path)
    backup_file = join(backup_path, csv_name[:-4] + "_bkp" + str(date.today()) + ".csv")
    df_updated.to_csv(backup_file, index=False)
    print("\ncsv recorded.\n")


# =========================
# CALLBACKS
# =========================
@app.callback(
    [Output("dummy_finish_work", "data")],
    [Input("button_finish_work", "n_clicks")],
    [State("state_store_df", "data"), State("input_finish_work", "value")],
)
def save_dataset(i_button_save_csv_nclicks, s_store_df, s_input_finish_work_value):
    if i_button_save_csv_nclicks > 0:
        df_updated = pd.read_json(s_store_df)

        if not exists(s_input_finish_work_value):
            mkdir(s_input_finish_work_value)

        # cria pastas
        for label in df_updated["manual_label"].astype(str).unique().tolist():
            out_dir = join(s_input_finish_work_value, label)
            if not exists(out_dir):
                mkdir(out_dir)

        # copia imagens (SEM prefixo "main/")
        for _, row in df_updated.iterrows():
            name = str(row["names"])
            folder = str(row["manual_label"])
            src = join(path_to_images, name)
            dst = join(s_input_finish_work_value, folder, name)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"[WARN] copy failed: {src} -> {dst} ({e})")

        print("\nDataset recorded.\n")

    return [0]


@app.callback(
    [Output("dummy_csv_save", "data")],
    [Input("button_save_csv", "n_clicks")],
    [State("state_store_df", "data"), State("input_save_csv", "value")],
)
def button_save_csv(i_button_save_csv_nclicks, s_store_df, s_input_save_csv_value):
    if i_button_save_csv_nclicks > 0:
        df_updated = pd.read_json(s_store_df)
        save_csv(df_updated, str(s_input_save_csv_value))
    return [0]


@app.callback(
    [
        Output("selected_custom_points", "data"),
        Output("state_store_df", "data"),
        Output("chart_flag", "data"),
        Output("unchecked_points", "data"),
    ],
    [
        Input("g_scatter_plot", "selectedData"),
        Input("g_coordenadas_paralelas", "restyleData"),
        Input("button_aplicar_novo_label", "n_clicks"),
        Input("button_invert_marks", "n_clicks"),
        Input("button_undo", "n_clicks"),
    ],
    [
        State("selected_custom_points", "data"),
        State("unchecked_points", "data"),
        State("state_store_df", "data"),
        State("g_coordenadas_paralelas", "figure"),
        State("g_image_selector", "images"),
        State("input_aplicar_novo_label", "value"),
        State("check_save_csv", "value"),
        State("input_save_csv", "value"),
        State("check_discard_marked", "value"),
    ],
)
def mudanca_custom_data(
    i_selection_g_scatter_plot,
    i_g_coordenadas_paralelas_restyleData,
    i_button_aplicar_novo_label_nclicks,
    i_invert_marks_nclicks,
    i_button_undo_nclicks,
    s_selected_custom_points,
    s_unchecked_points,
    s_store_df,
    s_g_coordenadas_paralelas_figure,
    s_button_galeria_filtrar_images,
    s_input_aplicar_novo_label_value,
    s_check_save_csv_value,
    s_input_save_csv_value,
    s_check_discard_marked_value,
):
    global labels_colors, next_label_id, labels_buffer, colors_buffer, init_par_coords

    init_par_coords = False
    set_chart_flag = 0

    df_updated = pd.read_json(s_store_df)
    ctx = dash.callback_context
    flag_callback = ctx.triggered[0]["prop_id"].split(".")[0]

    df_store_updated = df_updated.to_json()
    selectedpoints = s_selected_custom_points if s_selected_custom_points else json.dumps([])
    unchecked_points = []

    if flag_callback == "g_scatter_plot":
        set_chart_flag = 0
        if i_selection_g_scatter_plot and i_selection_g_scatter_plot.get("points"):
            selected = [d.get("customdata") for d in i_selection_g_scatter_plot["points"]]
            selected = [x for x in selected if x is not None]
            selectedpoints = json.dumps(selected)
        else:
            selectedpoints = json.dumps([])
        init_par_coords = True

    elif flag_callback == "g_coordenadas_paralelas":
        set_chart_flag = 1

        # se não tiver selectedData do scatter, usa o store
        if i_selection_g_scatter_plot and i_selection_g_scatter_plot.get("points"):
            selected = [d.get("customdata") for d in i_selection_g_scatter_plot["points"]]
            selected = [x for x in selected if x is not None]
        else:
            try:
                selected = json.loads(s_selected_custom_points) if s_selected_custom_points else []
            except Exception:
                selected = []

        filtered_df = df_updated[df_updated["custom_data"].isin(selected)]

        if i_g_coordenadas_paralelas_restyleData is not None and len(_columns_paralelas_coordenadas) > 0:
            selected_and_checked = update_df_paralelas_coord(
                _df=filtered_df,
                _list_columns=_columns_paralelas_coordenadas,
                _figure=s_g_coordenadas_paralelas_figure,
                _new_dim_vals=i_g_coordenadas_paralelas_restyleData,
            )

            if s_button_galeria_filtrar_images:
                for point in s_button_galeria_filtrar_images:
                    if point.get("custom_data") not in selected_and_checked:
                        unchecked_points.append(point.get("custom_data"))

            selectedpoints = json.dumps(selected)
        else:
            selectedpoints = json.dumps(list(df_updated["custom_data"]))

    elif flag_callback == "button_aplicar_novo_label":
        # marcações vêm da galeria
        selected_and_checked = []
        selected_not_checked = []

        if s_button_galeria_filtrar_images:
            for point in s_button_galeria_filtrar_images:
                cd = point.get("custom_data")
                if cd is None:
                    continue
                if point.get("isSelected") is True:
                    selected_and_checked.append(cd)
                else:
                    selected_not_checked.append(cd)

        new_label = str(s_input_aplicar_novo_label_value)

        # buffers para UNDO
        labels_buffer = df_updated["manual_label"].copy()
        colors_buffer = df_updated["colors"].copy()

        # aplica label
        df_updated.loc[df_updated["custom_data"].isin(selected_and_checked), "manual_label"] = new_label

        if new_label not in labels_colors:
            labels_colors[new_label] = next_label_id
            next_label_id += 1

        df_updated.loc[df_updated["custom_data"].isin(selected_and_checked), "colors"] = labels_colors[new_label]

        if s_check_save_csv_value == [" Save CSV after labeling"]:
            save_csv(df_updated, str(s_input_save_csv_value))

        df_store_updated = df_updated.to_json()

        if s_check_discard_marked_value == [" Discard images after labeling"]:
            selectedpoints = json.dumps(selected_not_checked)

    elif flag_callback == "button_undo":
        if labels_buffer is not None and colors_buffer is not None:
            df_updated["manual_label"] = labels_buffer
            df_updated["colors"] = colors_buffer
            df_store_updated = df_updated.to_json()
        selectedpoints = json.dumps([])

    elif flag_callback == "button_invert_marks":
        if s_button_galeria_filtrar_images:
            for point in s_button_galeria_filtrar_images:
                if point.get("isSelected") is True:
                    unchecked_points.append(point.get("custom_data"))
        df_store_updated = df_updated.to_json()

    s_unchecked_points = json.dumps([x for x in unchecked_points if x is not None])
    return [selectedpoints, df_store_updated, set_chart_flag, s_unchecked_points]


@app.callback(
    [
        Output("g_scatter_plot", "figure"),
        Output("g_image_selector", "images"),
        Output("g_coordenadas_paralelas", "figure"),
    ],
    [
        Input("selected_custom_points", "data"),
        Input("unchecked_points", "data"),
        Input("dropdown_order_labels", "value"),
        Input("slider_map_opacity", "value"),
        Input("slider_marker_size", "value"),
        Input("probs_range_slider", "value"),
        Input("vars_range_slider", "value"),
        Input("dropdown_order_images", "value"),
        Input("check_marked_first", "value"),
        Input("check_discard_relabeled", "value"),
    ],
    [
        State("g_scatter_plot", "figure"),
        State("g_image_selector", "images"),
        State("g_coordenadas_paralelas", "figure"),
        State("state_store_df", "data"),
        State("chart_flag", "data"),
    ],
)
def scatter_plot_image_selector(
    i_selected_custom_points,
    i_unchecked_points,
    i_dropdown_order_labels_value,
    i_slider_map_opacity_value,
    i_slider_marker_size_value,
    i_range_slider_probs,
    i_range_slider_vars,
    i_dropdown_order_images_value,
    i_check_marked_first_value,
    i_check_discard_relabeled_value,
    s_g_scatter_plot_figure,
    s_g_image_selector_images,
    s_g_coordenadas_paralelas_figure,
    s_store_df,
    s_chart_flag_data,
):
    global background_path, init_par_coords, thumbs_jpg

    prev_fig = go.Figure(s_g_scatter_plot_figure)
    ctx = dash.callback_context
    flag_callback = ctx.triggered[0]["prop_id"].split(".")[0]

    _df = pd.read_json(s_store_df)

    if flag_callback in [
        "selected_custom_points",
        "dropdown_order_labels",
        "slider_map_opacity",
        "slider_marker_size",
        "probs_range_slider",
        "vars_range_slider",
        "dropdown_order_images",
        "check_marked_first",
        "check_discard_relabeled",
    ]:
        opacity_changed = (flag_callback == "slider_map_opacity")

        unchecked_points = json.loads(i_unchecked_points) if i_unchecked_points else []
        unchecked_set = set(unchecked_points)

        selected_points = json.loads(i_selected_custom_points) if i_selected_custom_points else []
        if init_par_coords:
            init_for_update_pc(selected_points)

                # ---------------------------
        # ORDEM DAS LABELS NO PLOT
        # ---------------------------
        label_order = None
        if "manual_label" in _df.columns:
            labs = _df["manual_label"].astype(str)
            if i_dropdown_order_labels_value == "Frequency":
                # mais frequentes primeiro
                label_order = labs.value_counts().index.tolist()
            else:
                # A-Z
                label_order = sorted(labs.unique().tolist(), key=lambda s: s.lower())

        fig_scatter = f_figure_scatter_plot(
            _df,
            _columns=["x", "y"],
            _selected_custom_data=selected_points,
            prev_fig=prev_fig,
            order_by=i_dropdown_order_labels_value,
            background_img=background_path if (background_path and exists(background_path)) else None,
            opacity_changed=opacity_changed,
            opacity=i_slider_map_opacity_value,
            marker_size=i_slider_marker_size_value,
            xrange=background_ranges["x"],
            yrange=background_ranges["y"],
        )

        # ✅ Força a ordem na legenda/traces
        if label_order is not None:
            try:
                data_sorted = sorted(
                    list(fig_scatter.data),
                    key=lambda tr: label_order.index(str(tr.name)) if str(tr.name) in label_order else 10**9,
                )
                fig_scatter.data = tuple(data_sorted)
            except Exception:
                pass


        filtered_df = _df.loc[_df["custom_data"].isin(selected_points)]

        # filtro confiança
        if "confs1" in filtered_df.columns and i_range_slider_probs is not None:
            filtered_df = filtered_df.loc[filtered_df["confs1"] >= i_range_slider_probs[0]]
            filtered_df = filtered_df.loc[filtered_df["confs1"] <= i_range_slider_probs[1]]

        # filtro variância
        if "var_norm" in filtered_df.columns and i_range_slider_vars is not None:
            filtered_df = filtered_df.loc[filtered_df["var_norm"] >= i_range_slider_vars[0]]
            filtered_df = filtered_df.loc[filtered_df["var_norm"] <= i_range_slider_vars[1]]

        if i_check_discard_relabeled_value == [" Hide relabeled images"]:
            filtered_df = filtered_df[filtered_df["manual_label"] == filtered_df["correct_label"]]

        # ordenação multi-critério
        order_vals = i_dropdown_order_images_value
        if isinstance(order_vals, str):
            order_vals = [order_vals]
        if not order_vals:
            order_vals = ["A-Z, a-z"]

        sort_keys, ascending = [], []
        for crit in order_vals:
            if crit in map_columns_vals.values():
                key = [k for k, v in map_columns_vals.items() if v == crit][0]
                if key in filtered_df.columns:
                    sort_keys.append(key)
                    ascending.append(False)
                continue

            if crit == "Similarity":
                if "D7" in filtered_df.columns:
                    sort_keys.append("D7")
                    ascending.append(True)
                continue

            if crit == "Confidence":
                if "confs1" in filtered_df.columns:
                    sort_keys.append("confs1")
                    ascending.append(False)
                continue

            if crit == "Variance":
                if "var_norm" in filtered_df.columns:
                    sort_keys.append("var_norm")
                    ascending.append(False)
                continue

            if crit == "Conf diff":
                if ("confs1" in filtered_df.columns) and ("confs2" in filtered_df.columns):
                    if "conf_diff" not in filtered_df.columns:
                        filtered_df["conf_diff"] = filtered_df["confs1"] - filtered_df["confs2"]
                    sort_keys.append("conf_diff")
                    ascending.append(False)
                continue

            if crit == "A-Z, a-z":
                if "manual_label" in filtered_df.columns:
                    sort_keys.append("manual_label")
                    ascending.append(True)
                continue

            if crit in filtered_df.columns:
                sort_keys.append(crit)
                ascending.append(True)

        if len(sort_keys) == 0:
            sort_keys = ["manual_label"] if "manual_label" in filtered_df.columns else ["names"]
            ascending = [True]

        ordered_df = filtered_df.sort_values(by=sort_keys, ascending=ascending, kind="mergesort")

        # preserva seleção/unchecked
        gallery_map = {}
        if s_g_image_selector_images:
            for it in s_g_image_selector_images:
                cd = it.get("custom_data")
                if cd is not None:
                    gallery_map[cd] = bool(it.get("isSelected", True))

        data_temp = selected_points
        selection_temp = []
        for cd in data_temp:
            if cd in unchecked_set:
                selection_temp.append(False)
            else:
                selection_temp.append(gallery_map.get(cd, True))

        df_selection = pd.DataFrame({"custom_data": data_temp, "selected": selection_temp})
        ordered_df = pd.merge(ordered_df, df_selection, how="inner", on="custom_data")

        if i_check_marked_first_value == [" Marked images first"]:
            checked_df = ordered_df[ordered_df["selected"] == True]
            unchecked_df = ordered_df[ordered_df["selected"] == False]
            ordered_df = pd.concat([checked_df, unchecked_df])

        _image_correct = ordered_df["correct_label"]
        _image_names = ordered_df["names"]

        if thumbs_jpg:
            _image_thumbs = ordered_df["thumbnails"]
        else:
            _image_thumbs = ordered_df["names"]

        _image_widths = ordered_df["widths"]
        _image_heights = ordered_df["heights"]

        # atributo do caption (primeiro critério, se for métrica)
        _image_attributes = [""] * ordered_df.shape[0]
        first_crit = order_vals[0] if len(order_vals) > 0 else ""
        if first_crit in map_columns_vals.values():
            key = [k for k, v in map_columns_vals.items() if v == first_crit][0]
            if key in ordered_df.columns:
                _image_attributes = ordered_df[key].tolist()

        if "preds1" in ordered_df.columns:
            p1 = ordered_df["preds1"]
            c1 = ordered_df["confs1"]

        if "preds2" in ordered_df.columns:
            p2 = ordered_df["preds2"]
            c2 = ordered_df["confs2"]
            p3 = ordered_df["preds3"]
            c3 = ordered_df["confs3"]

        if "preds1" not in ordered_df.columns:
            texts = [f"{name[:-4]} - Pred: {lab}, Attr: {val}" for name, lab, val in zip(_image_names, _image_correct, _image_attributes)]
        elif "preds2" not in ordered_df.columns:
            texts = [f"{name[:-4]} - {pp1} ({100*cc1:.1f}%, Attr: {val})" for pp1, cc1, name, val in zip(p1, c1, _image_names, _image_attributes)]
        else:
            texts = [
                f"{name[:-4]} - {pp1} ({100*cc1:.1f}%, Attr: {val}), {pp2} ({100*cc2:.1f}%), {pp3} ({100*cc3:.1f}%)"
                for pp1, cc1, pp2, cc2, pp3, cc3, name, val in zip(p1, c1, p2, c2, p3, c3, _image_names, _image_attributes)
            ]

        fig2 = create_list_dics(
            _list_src=[join(path_to_images, n) for n in _image_names.tolist()],
            _list_thumbnail=[join(path_to_thumbnails, t) for t in _image_thumbs.tolist()],
            _list_name_figure=_image_names.tolist(),
            _list_thumbnailWidth=_image_widths.tolist(),
            _list_thumbnailHeight=_image_heights.tolist(),
            _list_isSelected=list(ordered_df["selected"]),
            _list_custom_data=ordered_df["custom_data"].tolist(),
            _list_caption=texts,
            _list_thumbnailCaption=ordered_df["manual_label"].tolist(),
            _list_tags=[[]] * ordered_df.shape[0],
        )

        # ParCoords: se não há colunas numéricas, não chama (evita KeyError)
        if len(_columns_paralelas_coordenadas) == 0:
            fig3 = go.Figure(s_g_coordenadas_paralelas_figure)
        else:
            if s_chart_flag_data == 1:
                fig3 = f_figure_paralelas_coordenadas(
                    _df=_df,
                    _filtered_df=filtered_df,
                    _columns=_columns_paralelas_coordenadas,
                    _selected_custom_data=selected_points,
                    _fig=s_g_coordenadas_paralelas_figure,
                )
            else:
                fig3 = f_figure_paralelas_coordenadas(
                    _df=_df,
                    _filtered_df=filtered_df,
                    _columns=_columns_paralelas_coordenadas,
                    _selected_custom_data=selected_points,
                    _fig=None,
                )

        return [fig_scatter, fig2, fig3]

    return [s_g_scatter_plot_figure, s_g_image_selector_images, s_g_coordenadas_paralelas_figure]


# =========================
# RUN
# =========================
_opened = False
if not _opened:
    webbrowser.open("http://127.0.0.1:" + str(port) + "/", new=2, autoraise=True)
    _opened = True

if __name__ == "__main__":
    app.title = "IAT"
    app.run_server(debug=False, port=port)

