# app.py
import sys
import re
import json
import shutil
import webbrowser
import tempfile
import uuid
from bisect import bisect_right
from datetime import date
from os import mkdir, listdir
from os.path import exists, join, dirname, basename, isdir, splitext
from io import StringIO
from urllib.parse import quote

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from PIL import Image as PILImage

from functions import *

import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import imageselector

# --- clustering (subset-based) ---
try:
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
except Exception:
    linkage = None
    fcluster = None
    dendrogram = None

try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None


# =========================
# CONFIG
# =========================
FIXED_BACKGROUND_OPACITY = 0.35
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# O runner.py roda com cwd=/mnt/md0/IAT, mas as imagens estão em /mnt/md0/IAT/main/assets/...
# Então forçamos o Dash a servir assets a partir do diretório correto:
ASSETS_FOLDER = join(dirname(__file__), "assets")  # /mnt/md0/IAT/main/assets
ASSETS_URL_PATH = "/assets"

BLANK_BG_PATH = join(tempfile.gettempdir(), "iat_blank_bg.png")

OVERVIEW_VALUE = "__OVERVIEW__"

# limites (segurança/performance)
MAX_CLUSTER_N = 2500
MAX_OVERVIEW_POINTS_PER_CLASS = 20000
MAX_OVERVIEW_SELECTOR_TOTAL = 120000   # pontos invisíveis (para capturar lassoPoints/range)
MAX_ALLB_PAGE_SIZE = 8000

# caches server-side (mantêm mudanças não salvas)
BATCH_DF_CACHE: dict[str, pd.DataFrame] = {}       # key: csv_basename -> df MODIFICADO/aberto
CSV_COLS_CACHE: dict[str, list[str]] = {}          # csv_basename -> cols
BATCH_BBOX_CACHE: dict[str, tuple[float, float, float, float]] = {}  # csv_basename -> (xmin,xmax,ymin,ymax)

OVERVIEW_CACHE = {"project": None, "version": None, "fig": None}

# ALL-BATCHES selection cache (máscaras compactas por batch)
ALLB_CACHE: dict[str, dict] = {}  # token -> {spec, batch_order, counts, cum_counts, packmasks, lens, index_map, version, ...}

# dirty batches (modificados em memória)
DIRTY_BATCHES: set[str] = set()

# undo (última operação)
UNDO_SNAPSHOT = None  # dict


# =========================
# SMALL UTILS
# =========================
def _bg_or_blank(bg_path: str | None) -> str:
    """Garante que background_img passado para functions.py (PIL.Image.open) nunca seja None."""
    if isinstance(bg_path, str) and bg_path and exists(bg_path):
        return bg_path

    if not exists(BLANK_BG_PATH):
        PILImage.new("RGBA", (4, 4), (0, 0, 0, 0)).save(BLANK_BG_PATH)

    return BLANK_BG_PATH


def _dir_has_images(p: str) -> bool:
    try:
        return any(f.lower().endswith(IMAGE_EXTS) for f in listdir(p))
    except Exception:
        return False


def _count_images(p: str) -> int:
    try:
        return sum(1 for f in listdir(p) if f.lower().endswith(IMAGE_EXTS))
    except Exception:
        return 0


def _ensure_samples_dir(p: str) -> str:
    """
    Se p não contiver imagens mas tiver subpasta samples, retorna p/samples.
    Se p NÃO existir e terminar em /samples, tenta voltar pro pai.
    """
    if not p:
        return p

    # Se o caminho não existe, mas termina em /samples, tenta o diretório pai
    if not isdir(p):
        base = basename(p).lower()
        parent = dirname(p)
        if base in ("samples", "sample") and isdir(parent):
            return parent
        return p

    # Se já tem imagens, ok
    if _dir_has_images(p):
        return p

    # Se é uma pasta "samples" mas não tem imagens, tenta o pai
    base = basename(p).lower()
    parent = dirname(p)
    if base in ("samples", "sample") and isdir(parent) and _dir_has_images(parent):
        return parent

    # Procura subpasta samples
    for sub in ("samples", "Samples", "SAMPLES", "sample", "Sample"):
        sp = join(p, sub)
        if isdir(sp) and _dir_has_images(sp):
            return sp

    # Se só existe uma subpasta, entra nela
    subs = [d for d in listdir(p) if isdir(join(p, d))]
    if len(subs) == 1:
        only = join(p, subs[0])
        return only

    return p


def _abs_to_assets_url(p: str) -> str:
    """
    Converte path ABSOLUTO dentro de ASSETS_FOLDER em URL /assets/...
    Isso é necessário para o navegador carregar a imagem.
    """
    if not p:
        return p

    p_norm = p.replace("\\", "/")
    root = ASSETS_FOLDER.replace("\\", "/").rstrip("/")

    # dentro do assets_folder
    if p_norm.startswith(root + "/"):
        rel = p_norm[len(root) + 1 :]
        rel_enc = "/".join(quote(seg) for seg in rel.split("/"))
        return f"{ASSETS_URL_PATH}/{rel_enc}"

    # fallback: se já veio relativo
    if p_norm.startswith("assets/"):
        rel = p_norm[len("assets/") :]
        rel_enc = "/".join(quote(seg) for seg in rel.split("/"))
        return f"{ASSETS_URL_PATH}/{rel_enc}"

    # fallback: path relativo qualquer
    if not p_norm.startswith("/"):
        rel_enc = "/".join(quote(seg) for seg in p_norm.split("/"))
        return f"{ASSETS_URL_PATH}/{rel_enc}"

    # absolute fora do assets -> provavelmente não vai funcionar, mas evita crash
    return p_norm


def _new_token() -> str:
    return uuid.uuid4().hex


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


# =========================
# PATH / BATCH HELPERS
# =========================
def _infer_batch_folder_from_csv_basename(csv_basename_no_ext: str):
    m = re.match(r"^(batch_\d+)", csv_basename_no_ext)
    return m.group(1) if m else None


def _replace_batch_in_path(template_path: str, new_batch_folder: str) -> str:
    """Substitui o primeiro segmento batch_#### encontrado no path por new_batch_folder."""
    parts = [p for p in template_path.replace("\\", "/").split("/") if p]
    for i, p in enumerate(parts):
        if re.match(r"^batch_\d+$", p):
            parts[i] = new_batch_folder
            return ("/" + "/".join(parts)) if template_path.startswith("/") else "/".join(parts)
    return template_path


def _resolve_background(project_name: str, batch_str: str):
    """Procura em main/assets/<project>/backgrounds/ arquivo cujo stem == batch_str."""
    background_base = join(ASSETS_FOLDER, project_name, "backgrounds")
    if not isdir(background_base):
        return None
    matches = [f for f in listdir(background_base) if splitext(f)[0] == batch_str]
    if matches:
        return join(background_base, matches[0])
    return None


def _pick_first_csv(csv_folder: str) -> str:
    if not isdir(csv_folder):
        raise AssertionError(f"Pasta de CSV não existe: {csv_folder}")
    cands = sorted([f for f in listdir(csv_folder) if f.lower().endswith(".csv") and not f.startswith(".")])
    if not cands:
        raise AssertionError(f"Nenhum CSV encontrado em: {csv_folder}")
    return join(csv_folder, cands[0])


def _list_csv_candidates(csv_folder: str):
    if not isdir(csv_folder):
        return []
    return sorted([f for f in listdir(csv_folder) if f.lower().endswith(".csv") and not f.startswith(".")])


# =========================
# CSV COLUMN / MINIMAL READ HELPERS (para não estourar RAM em projetos enormes)
# =========================
def _get_csv_columns(csv_path: str, csv_basename: str) -> list[str]:
    if csv_basename in CSV_COLS_CACHE:
        return CSV_COLS_CACHE[csv_basename]
    try:
        cols = list(pd.read_csv(csv_path, nrows=0, encoding="ISO-8859-1").columns)
    except Exception:
        cols = []
    CSV_COLS_CACHE[csv_basename] = cols
    return cols


def _read_batch_minimal(csv_path: str, csv_basename: str) -> pd.DataFrame:
    """
    Lê colunas mínimas necessárias para:
      - overview (x,y,manual_label)
      - all-batches selection (x,y,manual_label,correct_label,confs1,var_norm,custom_data,names)
    Importante: NÃO coloca em cache (a menos que o batch já esteja cacheado por ter sido modificado/aberto).
    """
    # se já cacheado (e pode ter mudanças), use cache
    if csv_basename in BATCH_DF_CACHE:
        df = BATCH_DF_CACHE[csv_basename].copy()
        return df

    cols = _get_csv_columns(csv_path, csv_basename)
    want = ["names", "custom_data", "manual_label", "correct_label", "x", "y", "confs1", "var_norm", "colors"]
    usecols = [c for c in want if c in cols]
    if not usecols:
        # fallback: tenta ler tudo (último caso)
        df = pd.read_csv(csv_path, encoding="ISO-8859-1", low_memory=False)
    else:
        df = pd.read_csv(csv_path, encoding="ISO-8859-1", usecols=usecols, low_memory=False)

    # normaliza o mínimo (sem criar colunas pesadas desnecessárias)
    if "names" not in df.columns:
        raise ValueError(f"{csv_basename}: CSV sem coluna 'names' (obrigatória).")

    if "manual_label" not in df.columns:
        if "correct_label" in df.columns:
            df["manual_label"] = df["correct_label"].astype(str)
        else:
            df["manual_label"] = "unlabeled"
    if "correct_label" not in df.columns:
        df["correct_label"] = df["manual_label"].astype(str)

    if "custom_data" not in df.columns:
        df["custom_data"] = df["names"].astype(str)
    else:
        df["custom_data"] = df["custom_data"].astype(str)

    df["names"] = df["names"].astype(str)
    df["manual_label"] = df["manual_label"].astype(str)
    df["correct_label"] = df["correct_label"].astype(str)

    if "x" in df.columns:
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
    if "y" in df.columns:
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

    if "confs1" in df.columns:
        df["confs1"] = pd.to_numeric(df["confs1"], errors="coerce")
    if "var_norm" in df.columns:
        df["var_norm"] = pd.to_numeric(df["var_norm"], errors="coerce")

    return df


def _compute_batch_bbox(csv_path: str, csv_basename: str) -> tuple[float, float, float, float] | None:
    """
    bbox para prefiltrar batches no all-batches selection.
    Cacheado. Usa cache de DF se existir; senão lê só x,y.
    """
    if csv_basename in BATCH_BBOX_CACHE:
        return BATCH_BBOX_CACHE[csv_basename]

    try:
        if csv_basename in BATCH_DF_CACHE:
            df = BATCH_DF_CACHE[csv_basename]
            x = pd.to_numeric(df.get("x", pd.Series([], dtype=float)), errors="coerce")
            y = pd.to_numeric(df.get("y", pd.Series([], dtype=float)), errors="coerce")
        else:
            cols = _get_csv_columns(csv_path, csv_basename)
            usecols = [c for c in ["x", "y"] if c in cols]
            if not usecols:
                return None
            tmp = pd.read_csv(csv_path, encoding="ISO-8859-1", usecols=usecols, low_memory=False)
            x = pd.to_numeric(tmp.get("x", pd.Series([], dtype=float)), errors="coerce")
            y = pd.to_numeric(tmp.get("y", pd.Series([], dtype=float)), errors="coerce")

        x = x.dropna()
        y = y.dropna()
        if len(x) == 0 or len(y) == 0:
            return None

        bb = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
        BATCH_BBOX_CACHE[csv_basename] = bb
        return bb
    except Exception:
        return None


def _bbox_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _poly_bbox(poly_xy: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly_xy]
    ys = [p[1] for p in poly_xy]
    return (float(min(xs)), float(max(xs)), float(min(ys)), float(max(ys)))


def _points_in_poly(x: np.ndarray, y: np.ndarray, poly_xy: list[tuple[float, float]]) -> np.ndarray:
    """
    Ray casting (vetorizado por iteração em arestas).
    poly_xy: lista de (x,y) fechada ou não (se não fechada, fecha internamente).
    Retorna mask boolean.
    """
    if poly_xy is None or len(poly_xy) < 3:
        return np.zeros_like(x, dtype=bool)

    # garante float
    poly = [(float(px), float(py)) for px, py in poly_xy]
    if poly[0] != poly[-1]:
        poly.append(poly[0])

    px = np.array([p[0] for p in poly], dtype=np.float64)
    py = np.array([p[1] for p in poly], dtype=np.float64)

    n = len(poly) - 1
    inside = np.zeros(x.shape[0], dtype=bool)

    # evita divisão por zero
    eps = 1e-12

    for i in range(n):
        x1, y1 = px[i], py[i]
        x2, y2 = px[i + 1], py[i + 1]
        cond = ((y1 > y) != (y2 > y))
        x_int = (x2 - x1) * (y - y1) / (y2 - y1 + eps) + x1
        inside ^= (cond & (x < x_int))
    return inside


def _parse_selection_polygon(selectedData: dict | None) -> dict | None:
    """
    Extrai geometria de seleção do Plotly:
      - lassoPoints: polígono
      - range: caixa
    Retorna dict:
      {"type":"lasso","poly":[(x,y),...]} ou {"type":"box","range":{"x":[x0,x1],"y":[y0,y1]}}
    """
    if not selectedData or not isinstance(selectedData, dict):
        return None

    lp = selectedData.get("lassoPoints")
    if isinstance(lp, dict) and "x" in lp and "y" in lp and lp["x"] and lp["y"]:
        poly = list(zip(lp["x"], lp["y"]))
        if len(poly) >= 3:
            return {"type": "lasso", "poly": [(float(a), float(b)) for a, b in poly]}

    rg = selectedData.get("range")
    if isinstance(rg, dict) and "x" in rg and "y" in rg and rg["x"] and rg["y"]:
        try:
            x0, x1 = float(rg["x"][0]), float(rg["x"][1])
            y0, y1 = float(rg["y"][0]), float(rg["y"][1])
            return {"type": "box", "range": {"x": [min(x0, x1), max(x0, x1)], "y": [min(y0, y1), max(y0, y1)]}}
        except Exception:
            return None

    return None


def _global_id(csv_basename: str, custom_data_orig: str) -> str:
    return f"{csv_basename}::{custom_data_orig}"


def _split_global_id(gid: str) -> tuple[str, str] | None:
    if not isinstance(gid, str) or "::" not in gid:
        return None
    a, b = gid.split("::", 1)
    if not a or not b:
        return None
    return a, b


def _images_dir_for_csv(s_ctx: dict, csv_basename: str) -> str:
    """
    Dado csv_basename do batch, retorna diretório de imagens (FS) usando images_path_template do contexto.
    """
    images_template = str(s_ctx.get("images_path_template", s_ctx.get("path_to_images", "")))
    stem = csv_basename[:-4]
    batch_folder = _infer_batch_folder_from_csv_basename(stem)

    if batch_folder:
        p = _replace_batch_in_path(images_template, batch_folder)
    else:
        p = images_template

    return _ensure_samples_dir(p)


# =========================
# DATA HELPERS
# =========================
def read_input_csv(csv_file):
    return pd.read_csv(csv_file, encoding="ISO-8859-1")


def init_appearance():
    return "rgba(255, 250, 240, 100)"


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "names" not in df.columns:
        raise ValueError("CSV não possui coluna 'names' (obrigatória).")

    if "manual_label" not in df.columns:
        if "correct_label" in df.columns:
            df["manual_label"] = df["correct_label"].astype(str)
        else:
            df["manual_label"] = "unlabeled"

    if "correct_label" not in df.columns:
        df["correct_label"] = df["manual_label"].astype(str)

    if "colors" not in df.columns:
        uniq = sorted(df["manual_label"].astype(str).unique().tolist())
        lab2id = {lab: i for i, lab in enumerate(uniq)}
        df["colors"] = df["manual_label"].astype(str).map(lab2id).astype(int)
    else:
        df["colors"] = pd.to_numeric(df["colors"], errors="coerce").fillna(0).astype(int)

    if "custom_data" not in df.columns:
        df["custom_data"] = df["names"].astype(str)
        if df["custom_data"].duplicated().any():
            df["custom_data"] = [f"{n}__{i}" for i, n in enumerate(df["names"].astype(str))]
    else:
        df["custom_data"] = df["custom_data"].astype(str)
        if df["custom_data"].duplicated().any():
            df["custom_data"] = [f"{cd}__{i}" for i, cd in enumerate(df["custom_data"].astype(str))]

    # reforça dtype
    df["names"] = df["names"].astype(str)
    df["custom_data"] = df["custom_data"].astype(str)
    df["manual_label"] = df["manual_label"].astype(str)
    df["correct_label"] = df["correct_label"].astype(str)

    # x,y precisam existir para o scatter
    if "x" in df.columns:
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
    if "y" in df.columns:
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

    return df


def _init_labels_colors(df: pd.DataFrame):
    labels_colors = {}
    for lab in df["manual_label"].astype(str).unique().tolist():
        labels_colors[lab] = int(df.loc[df["manual_label"].astype(str) == lab, "colors"].iloc[0])
    next_label_id = int(df["colors"].max()) + 1 if "colors" in df.columns and len(df) else 0
    return labels_colors, next_label_id


def _compute_dropdown_image_vals(df: pd.DataFrame, map_columns_vals: dict):
    dropdown_image_vals = ["A-Z, a-z", "Similarity"]

    if "confs1" in df.columns:
        dropdown_image_vals.append("Confidence")
        if "confs2" in df.columns:
            dropdown_image_vals.append("Conf diff")

    if "var_norm" in df.columns:
        dropdown_image_vals.append("Variance")

    for key in map_columns_vals.keys():
        if key in df.columns:
            dropdown_image_vals.append(map_columns_vals[key])

    return sorted(set(dropdown_image_vals))


def _load_batch_df_cached(csv_path: str, csv_basename: str) -> pd.DataFrame:
    """
    Carrega df de um batch usando cache server-side (somente quando batch foi aberto/alterado).
    """
    if csv_basename in BATCH_DF_CACHE:
        return BATCH_DF_CACHE[csv_basename].copy()

    df0 = _ensure_required_columns(read_input_csv(csv_path))
    # NOTE: aqui NÃO calculamos widths/heights (caro). Isso fica para galeria/página.
    BATCH_DF_CACHE[csv_basename] = df0.copy()
    return df0


# =========================
# OVERVIEW (all batches)
# =========================
def _palette_rgb(i: int):
    """
    Paleta de cores consistente com o que é usado nos batches.
    MODIFICAÇÃO: Paleta expandida para melhor consistência.
    """
    # Paleta Plotly padrão expandida (10 cores originais + 10 extras)
    pal = [
        (31, 119, 180),    # Azul
        (255, 127, 14),    # Laranja
        (44, 160, 44),     # Verde
        (214, 39, 40),     # Vermelho
        (148, 103, 189),   # Roxo
        (140, 86, 75),     # Marrom
        (227, 119, 194),   # Rosa
        (127, 127, 127),   # Cinza
        (188, 189, 34),    # Lima
        (23, 190, 207),    # Ciano
        
        # Cores adicionais para mais classes
        (174, 199, 232),   # Azul claro
        (255, 187, 120),   # Laranja claro
        (152, 223, 138),   # Verde claro
        (255, 152, 150),   # Vermelho claro
        (197, 176, 213),   # Roxo claro
        (196, 156, 148),   # Marrom claro
        (247, 182, 210),   # Rosa claro
        (199, 199, 199),   # Cinza claro
        (219, 219, 141),   # Lima claro
        (158, 218, 229),   # Ciano claro
        
        # Mais cores se necessário
        (214, 97, 107),    # Vermelho escuro
        (86, 180, 233),    # Azul médio
        (230, 171, 2),     # Amarelo
        (0, 158, 115),     # Verde escuro
        (240, 228, 66),    # Amarelo claro
        (0, 114, 178),     # Azul escuro
        (213, 94, 0),      # Laranja escuro
        (204, 121, 167),   # Magenta
    ]
    return pal[i % len(pal)]
    

def _gather_overview_points(project_name: str, csv_folder: str):
    """
    Junta pontos (x,y) por classe para TODO o projeto.
    IMPORTANTE: não cacheia batches inteiros (evita estourar RAM). Só usa cache se batch já foi aberto/alterado.
    Limita MAX_OVERVIEW_POINTS_PER_CLASS por classe (amostragem).
    """
    csvs = _list_csv_candidates(csv_folder)
    buckets: dict[str, list[np.ndarray]] = {}

    rng = np.random.default_rng(0)

    def _cap_append(label: str, xy: np.ndarray):
        if xy.size == 0:
            return
        if label not in buckets:
            buckets[label] = [xy]
        else:
            buckets[label].append(xy)

        total = sum(a.shape[0] for a in buckets[label])
        if total > MAX_OVERVIEW_POINTS_PER_CLASS:
            all_xy = np.vstack(buckets[label])
            k = min(MAX_OVERVIEW_POINTS_PER_CLASS, all_xy.shape[0])
            idx = rng.choice(all_xy.shape[0], size=k, replace=False)
            buckets[label] = [all_xy[idx]]

    for f in csvs:
        if f == OVERVIEW_VALUE:
            continue
        p = join(csv_folder, f)
        if not exists(p):
            continue

        try:
            dfb = _read_batch_minimal(p, f)
        except Exception:
            continue

        if "x" not in dfb.columns or "y" not in dfb.columns or "manual_label" not in dfb.columns:
            continue

        tmp = dfb[["x", "y", "manual_label"]].copy()
        tmp["manual_label"] = tmp["manual_label"].astype(str)
        tmp["x"] = pd.to_numeric(tmp["x"], errors="coerce")
        tmp["y"] = pd.to_numeric(tmp["y"], errors="coerce")
        tmp = tmp.dropna(subset=["x", "y", "manual_label"])
        if tmp.empty:
            continue

        for lab, grp in tmp.groupby("manual_label", sort=False):
            xy = grp[["x", "y"]].to_numpy(dtype=float)
            _cap_append(str(lab), xy)

    out = {}
    for lab, parts in buckets.items():
        out[lab] = np.vstack(parts) if len(parts) > 1 else parts[0]
    return out

def _get_class_colors_from_batches(csv_folder: str, labels: list) -> dict:
    """
    Obtém o mapeamento de cores das classes a partir dos batches.
    Retorna dict {label: color_id}
    """
    color_mapping = {}
    csvs = _list_csv_candidates(csv_folder)
    
    for csv_file in csvs:
        if csv_file == OVERVIEW_VALUE:
            continue
            
        csv_path = join(csv_folder, csv_file)
        try:
            # Ler apenas algumas linhas para economizar
            df_sample = pd.read_csv(csv_path, usecols=['manual_label', 'colors'], nrows=5000, encoding="ISO-8859-1")
            
            for label in df_sample['manual_label'].astype(str).unique():
                if label in labels and label not in color_mapping:
                    # Pegar o primeiro color_id encontrado para esta label
                    color_id = int(df_sample.loc[df_sample['manual_label'].astype(str) == label, 'colors'].iloc[0])
                    color_mapping[label] = color_id
                    
            # Se já encontramos todas as cores, podemos parar
            if all(label in color_mapping for label in labels):
                break
                
        except Exception as e:
            print(f"[WARN] Erro ao ler cores de {csv_file}: {e}")
            continue
    
    # Preencher qualquer label faltante com valores sequenciais
    for i, label in enumerate(labels):
        if label not in color_mapping:
            color_mapping[label] = i
    
    return color_mapping
    
    
def _build_overview_figure(project_name: str, csv_folder: str, graph_range: int) -> go.Figure:
    """
    "Hull + densidade" por classe.
    MODIFICAÇÃO: Sincroniza heat hull e convex hull no mesmo legendgroup.
    """
    pts_by_label = _gather_overview_points(project_name, csv_folder)

    fig = go.Figure()
    fig.update_layout(
        height=740,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(range=[-graph_range, graph_range], zeroline=False),
        yaxis=dict(range=[-graph_range, graph_range], zeroline=False, scaleanchor="x", scaleratio=1),
        legend=dict(
            orientation="v", 
            y=1.0, 
            x=1.01,
            title="Classes",
            font=dict(size=10),
            groupclick="toggleitem",  # Permite alternar a visibilidade
        ),
        title="Overview (all batches): hull + densidade por classe",
        dragmode="lasso",
    )

    labels_sorted = sorted(pts_by_label.keys(), key=lambda s: s.lower())

    # --- Obter mapeamento de cores das classes a partir dos batches ---
    color_mapping = _get_class_colors_from_batches(csv_folder, labels_sorted)

    # --- densidade + hulls ---
    for i, lab in enumerate(labels_sorted):
        xy = pts_by_label.get(lab)
        if xy is None or xy.size == 0:
            continue

        x = xy[:, 0]
        y = xy[:, 1]

        # Obter color_id do mapeamento
        color_id = color_mapping.get(lab, i)
        
        # Usar a mesma paleta que é usada nos batches
        r, g, b = _palette_rgb(color_id)
        
        # Ajustar opacidades para melhor visualização
        c0 = f"rgba({r},{g},{b},0.00)"
        c1 = f"rgba({r},{g},{b},0.35)"

        # Heat hull (densidade) - MESMO LEGENDGROUP
        fig.add_trace(
            go.Histogram2dContour(
                x=x,
                y=y,
                contours=dict(coloring="fill", showlines=False),
                ncontours=10,
                showscale=False,
                colorscale=[(0.0, c0), (1.0, c1)],
                hoverinfo="skip",
                name=lab,  # MESMO NOME
                showlegend=False,  # NÃO aparece na legenda individualmente
                legendgroup=lab,  # GRUPO DA LEGENDA COMPARTILHADO
                opacity=1.0,
            )
        )

        # Convex hull ou bounding box - MESMO LEGENDGROUP
        if ConvexHull is not None and xy.shape[0] >= 3:
            try:
                hull = ConvexHull(xy)
                poly = xy[hull.vertices]
                poly = np.vstack([poly, poly[0]])
                fig.add_trace(
                    go.Scatter(
                        x=poly[:, 0],
                        y=poly[:, 1],
                        mode="lines",
                        line=dict(width=1),
                        fill="toself",
                        fillcolor=f"rgba({r},{g},{b},0.05)",
                        name=lab,  # MESMO NOME
                        hoverinfo="skip",
                        showlegend=True,  # APENAS ESTE APARECE NA LEGENDA
                        legendgroup=lab,  # GRUPO DA LEGENDA COMPARTILHADO
                        visible=True,
                    )
                )
            except Exception:
                xmin, xmax = float(np.min(x)), float(np.max(x))
                ymin, ymax = float(np.min(y)), float(np.max(y))
                fig.add_trace(
                    go.Scatter(
                        x=[xmin, xmax, xmax, xmin, xmin],
                        y=[ymin, ymin, ymax, ymax, ymin],
                        mode="lines",
                        fill="toself",
                        fillcolor=f"rgba({r},{g},{b},0.04)",
                        name=lab,  # MESMO NOME
                        hoverinfo="skip",
                        showlegend=True,  # APENAS ESTE APARECE NA LEGENDA
                        legendgroup=lab,  # GRUPO DA LEGENDA COMPARTILHADO
                        visible=True,
                    )
                )
        else:
            # Linha vazia apenas para a legenda
            fig.add_trace(
                go.Scatter(
                    x=[], y=[],
                    mode="lines",
                    line=dict(width=1, color=f"rgb({r},{g},{b})"),
                    name=lab,
                    showlegend=True,
                    legendgroup=lab,  # GRUPO DA LEGENDA COMPARTILHADO
                )
            )

    # --- camada invisível para seleção (amostra global) ---
    rng = np.random.default_rng(123)
    
    for lab in labels_sorted:
        xy = pts_by_label.get(lab)
        if xy is None or xy.size == 0:
            continue
        # amostra por classe para cobrir bem o espaço
        k = min(1500, xy.shape[0])
        if k <= 0:
            continue
        idx = rng.choice(xy.shape[0], size=k, replace=False)
        sel = xy[idx]

        # Adicionar seletores por classe para filtrar por classe visível
        fig.add_trace(
            go.Scatter(
                x=sel[:, 0],
                y=sel[:, 1],
                mode="markers",
                marker=dict(size=6, color="rgba(0,0,0,0.003)"),
                hoverinfo="skip",
                name=lab,
                showlegend=False,
                legendgroup=lab,  # MESMO GRUPO da classe!
                selected=dict(marker=dict(color="rgba(0,0,0,0.003)")),
                unselected=dict(marker=dict(color="rgba(0,0,0,0.003)")),
                customdata=[[lab] for _ in range(k)],  # Todos com a mesma classe
                opacity=0.001,
                visible=True,  # Visibilidade controlada pelo grupo
            )
        )

    # Adicionar callback JavaScript para sincronizar visibilidade
    fig.update_layout(
        legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            groupclick="togglegroup",  # Alterna todo o grupo
        )
    )

    return fig
    
    
def _get_overview_fig_cached(project_name: str, csv_folder: str, version: int, graph_range: int) -> go.Figure:
    if (
        OVERVIEW_CACHE["fig"] is not None
        and OVERVIEW_CACHE["project"] == project_name
        and OVERVIEW_CACHE["version"] == version
    ):
        return OVERVIEW_CACHE["fig"]

    fig = _build_overview_figure(project_name, csv_folder, graph_range)
    
    fig.update_layout(uirevision=f"overview-{version}")
    
    OVERVIEW_CACHE["project"] = project_name
    OVERVIEW_CACHE["version"] = version
    OVERVIEW_CACHE["fig"] = fig
    return fig


# =========================
# ALL-BATCHES selection (máscaras compactas + paginação determinística)
# =========================
def _mask_for_df_by_spec(dfb: pd.DataFrame, spec: dict) -> np.ndarray:
    """
    Retorna mask boolean (len==len(dfb)) para os filtros do all-batches selection.
    MODIFICAÇÃO: Suporte a lista de classes no class_filter.
    """
    n = int(len(dfb))
    if n <= 0:
        return np.zeros((0,), dtype=bool)

    # x,y (obrigatório)
    if "x" not in dfb.columns or "y" not in dfb.columns:
        return np.zeros((n,), dtype=bool)

    x = pd.to_numeric(dfb["x"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(dfb["y"], errors="coerce").to_numpy(dtype=np.float64)

    ok = np.isfinite(x) & np.isfinite(y)

    geom = spec.get("geom", {})
    gtype = geom.get("type")

    if gtype == "lasso":
        poly = geom.get("poly", [])
        inside = _points_in_poly(x, y, poly)
        ok &= inside
    elif gtype == "box":
        rg = geom.get("range", {})
        xr = rg.get("x", None)
        yr = rg.get("y", None)
        if xr and yr:
            ok &= (x >= float(xr[0])) & (x <= float(xr[1])) & (y >= float(yr[0])) & (y <= float(yr[1]))
        else:
            return np.zeros((n,), dtype=bool)
    else:
        return np.zeros((n,), dtype=bool)

    # filtro por classe - agora pode ser None, string ou lista
    cls = spec.get("class_filter", None)
    if cls:
        if "manual_label" in dfb.columns:
            labs = dfb["manual_label"].astype(str).to_numpy(dtype=object)
            if isinstance(cls, str):
                ok &= (labs == str(cls))
            elif isinstance(cls, list) and len(cls) > 0:
                # Filtrar por múltiplas classes
                ok &= np.isin(labs, [str(c) for c in cls])
            # Se cls for uma lista vazia, não seleciona nada
            elif isinstance(cls, list) and len(cls) == 0:
                ok &= False
        else:
            return np.zeros((n,), dtype=bool)

    # conf range
    pr = spec.get("probs_rng", None)
    if pr is not None and "confs1" in dfb.columns:
        c1 = pd.to_numeric(dfb["confs1"], errors="coerce").to_numpy(dtype=np.float64)
        ok &= np.isfinite(c1) & (c1 >= float(pr[0])) & (c1 <= float(pr[1]))

    # var range
    vr = spec.get("vars_rng", None)
    if vr is not None and "var_norm" in dfb.columns:
        v1 = pd.to_numeric(dfb["var_norm"], errors="coerce").to_numpy(dtype=np.float64)
        ok &= np.isfinite(v1) & (v1 >= float(vr[0])) & (v1 <= float(vr[1]))

    # hide relabeled
    if bool(spec.get("hide_relab", False)):
        if "manual_label" in dfb.columns and "correct_label" in dfb.columns:
            ml = dfb["manual_label"].astype(str).to_numpy(dtype=object)
            cl = dfb["correct_label"].astype(str).to_numpy(dtype=object)
            ok &= (ml == cl)

    return ok.astype(bool, copy=False)
    
    
def _filter_selected_points_by_visible_classes(selected_data: dict, figure_data: dict) -> dict:
    """
    Filtra os pontos selecionados para incluir apenas aqueles cujas classes estão visíveis.
    
    Args:
        selected_data: Dados de seleção do Plotly
        figure_data: Figura atual do gráfico
    
    Returns:
        Dados de seleção filtrados (apenas pontos de classes visíveis)
    """
    if not selected_data or not figure_data or "data" not in figure_data:
        return selected_data
    
    # Extrair classes visíveis da figura
    visible_classes = []
    for trace in figure_data["data"]:
        trace_name = trace.get("name", "")
        trace_visible = trace.get("visible", True)
        
        # Verificar se é uma classe (não o seletor) e se está visível
        if trace_name and trace_name != "__selector__":
            if trace_visible is True:  # Apenas True, não 'legendonly' ou False
                visible_classes.append(trace_name)
    
    # Se não há classes visíveis ou todas estão visíveis, retorna sem filtrar
    if not visible_classes or "points" not in selected_data:
        return selected_data
    
    # Filtrar pontos selecionados
    filtered_points = []
    for point in selected_data["points"]:
        # O customdata do ponto contém a classe (conforme definido em _build_overview_figure)
        point_customdata = point.get("customdata", [])
        if point_customdata and len(point_customdata) > 0:
            point_class = point_customdata[0]  # A classe está no primeiro elemento do customdata
            if point_class in visible_classes:
                filtered_points.append(point)
    
    # Criar nova seleção filtrada
    if filtered_points:
        new_selected_data = selected_data.copy()
        new_selected_data["points"] = filtered_points
        
        # Atualizar também os arrays de pontos se existirem
        if "lassoPoints" in selected_data and selected_data["lassoPoints"]:
            # Para lasso, manter apenas os índices filtrados
            if "pointIndices" in selected_data:
                # Encontrar índices dos pontos filtrados
                filtered_indices = [point["pointIndex"] for point in filtered_points]
                new_selected_data["pointIndices"] = filtered_indices
        
        return new_selected_data
    
    # Se nenhum ponto foi filtrado, retornar seleção vazia
    return {"points": []}
    
    
def _get_visible_classes_from_figure(fig_dict: dict) -> list:
    """
    Extrai as classes visíveis da figura do Plotly.
    Verifica tanto os traces com showlegend=True quanto os com legendgroup.
    """
    if not fig_dict or "data" not in fig_dict:
        return []
    
    visible_classes = []
    trace_groups = {}
    
    for trace in fig_dict["data"]:
        trace_name = trace.get("name", "")
        trace_group = trace.get("legendgroup", "")
        trace_visible = trace.get("visible", True)
        
        # Se o trace tem um nome de classe (não o seletor)
        if trace_name and trace_name != "__selector__":
            # Se o trace tem um legendgroup, usar o grupo
            group_key = trace_group if trace_group else trace_name
            
            # Determinar se o grupo está visível
            if trace_visible is not False:  # True ou None
                if group_key not in trace_groups:
                    trace_groups[group_key] = trace_name
                elif trace_groups[group_key] != trace_name:
                    # Se já existe com nome diferente, manter o primeiro
                    pass
            else:
                # Se não está visível, remover do dicionário
                trace_groups.pop(group_key, None)
    
    # Coletar classes únicas visíveis
    visible_classes = list(set(trace_groups.values()))
    return visible_classes
    
    
def _allb_build_selection(project_name: str, csv_folder: str, csv_candidates: list[str], spec: dict, version: int, visible_classes: list = None) -> tuple[str, int]:
    """
    Constrói máscaras por batch (compactadas) e prefix-sums para paginação determinística.
    MODIFICAÇÃO: Adicionado parâmetro visible_classes para filtrar apenas classes visíveis na legenda.
    """
    batch_order = [f for f in csv_candidates if f != OVERVIEW_VALUE]
    token = _new_token()

    counts = [0] * len(batch_order)
    cum_counts = [0] * len(batch_order)
    packmasks: dict[str, bytes] = {}
    lens: dict[str, int] = {}
    index_map = {bn: i for i, bn in enumerate(batch_order)}

    # bbox para prefiltrar
    sel_bbox = None
    geom = spec.get("geom", {})
    if geom.get("type") == "lasso":
        sel_bbox = _poly_bbox(geom.get("poly", []))
    elif geom.get("type") == "box":
        xr = geom.get("range", {}).get("x", None)
        yr = geom.get("range", {}).get("y", None)
        if xr and yr:
            sel_bbox = (float(xr[0]), float(xr[1]), float(yr[0]), float(yr[1]))

    running = 0
    for i, bn in enumerate(batch_order):
        p = join(csv_folder, bn)
        if not exists(p):
            counts[i] = 0
            cum_counts[i] = running
            continue

        # bbox prefilter (se disponível)
        if sel_bbox is not None:
            bb = _compute_batch_bbox(p, bn)
            if bb is not None and (not _bbox_intersect(bb, sel_bbox)):
                counts[i] = 0
                cum_counts[i] = running
                continue

        try:
            dfb = _read_batch_minimal(p, bn)
        except Exception:
            counts[i] = 0
            cum_counts[i] = running
            continue
               
        if len(dfb) == 0:
            counts[i] = 0
            cum_counts[i] = running
            continue

        mask = _mask_for_df_by_spec(dfb, spec)
        n = int(len(mask))
        lens[bn] = n

        c = int(mask.sum()) if n else 0
        counts[i] = c
        running += c
        cum_counts[i] = running

        if c > 0:
            pack = np.packbits(mask.astype(np.uint8))
            packmasks[bn] = pack.tobytes()

    ALLB_CACHE[token] = dict(
        project=project_name,
        version=int(version),
        spec=spec,
        batch_order=batch_order,
        counts=counts,
        cum_counts=cum_counts,
        packmasks=packmasks,
        lens=lens,
        index_map=index_map,
        visible_classes=visible_classes,
    )

    return token, running


def _allb_recompute_batches(token: str, csv_folder: str, batch_names: list[str]) -> None:
    """
    Recalcula máscara de batches específicos, mantendo spec do token.
    Atualiza counts + cum_counts + packmasks.
    """
    cache = ALLB_CACHE.get(token)
    if not cache:
        return

    spec = cache.get("spec", {})
    batch_order = cache.get("batch_order", [])
    counts = cache.get("counts", [])
    packmasks = cache.get("packmasks", {})
    lens = cache.get("lens", {})
    index_map = cache.get("index_map", {})

    for bn in batch_names:
        if bn not in index_map:
            continue
        p = join(csv_folder, bn)
        if not exists(p):
            # zera
            idx = index_map[bn]
            counts[idx] = 0
            packmasks.pop(bn, None)
            lens[bn] = 0
            continue

        try:
            dfb = _read_batch_minimal(p, bn)
        except Exception:
            continue

        mask = _mask_for_df_by_spec(dfb, spec)
        n = int(len(mask))
        lens[bn] = n

        c = int(mask.sum()) if n else 0
        idx = index_map[bn]
        counts[idx] = c

        if c > 0:
            pack = np.packbits(mask.astype(np.uint8))
            packmasks[bn] = pack.tobytes()
        else:
            packmasks.pop(bn, None)

    # recomputa cum_counts
    running = 0
    cum = []
    for c in counts:
        running += int(c)
        cum.append(running)
    cache["cum_counts"] = cum
    cache["counts"] = counts
    cache["packmasks"] = packmasks
    cache["lens"] = lens


def _allb_total(token: str) -> int:
    cache = ALLB_CACHE.get(token)
    if not cache:
        return 0
    cum = cache.get("cum_counts", [])
    return int(cum[-1]) if cum else 0


# ===== COLE AQUI (logo abaixo de _allb_total) =====
def _allb_collect_selected_for_class(token: str, s_ctx: dict, class_label: str, unchecked_set: set[str] | None = None) -> pd.DataFrame:
    """
    Retorna um DF com TODOS os pontos selecionados pelo token ALLB_CACHE, já filtrando por class_label,
    e opcionalmente removendo itens desmarcados (unchecked_set) via global_id.
    """
    cache = ALLB_CACHE.get(token)
    if not cache:
        return pd.DataFrame()

    _csv_folder = str(s_ctx.get("csv_folder", ""))
    batch_order = cache.get("batch_order", [])
    packmasks = cache.get("packmasks", {})
    lens = cache.get("lens", {})

    rows = []
    for bn in batch_order:
        pack = packmasks.get(bn, None)
        n = int(lens.get(bn, 0))
        if pack is None or n <= 0:
            continue

        mask_bits = np.unpackbits(np.frombuffer(pack, dtype=np.uint8), count=n).astype(bool)
        idx_sel = np.flatnonzero(mask_bits)
        if idx_sel.size == 0:
            continue

        csv_path = join(_csv_folder, bn)
        if not exists(csv_path):
            continue

        dfb = _read_batch_minimal(csv_path, bn)
        if dfb is None or dfb.empty:
            continue

        idx_sel = idx_sel[idx_sel < len(dfb)]
        if idx_sel.size == 0:
            continue

        part = dfb.iloc[idx_sel].copy()

        if "manual_label" in part.columns:
            part = part[part["manual_label"].astype(str) == str(class_label)].copy()
        if part.empty:
            continue

        part["csv_basename"] = bn
        part["custom_data_orig"] = part["custom_data"].astype(str)
        part["custom_data"] = [_global_id(bn, cd) for cd in part["custom_data_orig"].astype(str).tolist()]

        if unchecked_set:
            part = part[~part["custom_data"].astype(str).isin(unchecked_set)].copy()
            if part.empty:
                continue

        rows.append(part)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    for c in ["names", "custom_data", "custom_data_orig", "manual_label", "correct_label", "csv_basename"]:
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out


def _read_batch_full_for_page(csv_path: str, csv_basename: str) -> pd.DataFrame:
    """
    Para montar página: se batch está no cache (modificado/aberto), usa ele; senão lê do disco (full).
    """
    if csv_basename in BATCH_DF_CACHE:
        return BATCH_DF_CACHE[csv_basename].copy()
    df = _ensure_required_columns(read_input_csv(csv_path))
    return df


def _allb_make_page_df(token: str, page: int, page_size: int, s_ctx: dict) -> pd.DataFrame:
    cache = ALLB_CACHE.get(token)
    if not cache:
        return pd.DataFrame()

    batch_order = cache.get("batch_order", [])
    counts = cache.get("counts", [])
    cum = cache.get("cum_counts", [])
    packmasks = cache.get("packmasks", {})
    lens = cache.get("lens", {})

    total = int(cum[-1]) if cum else 0
    if total <= 0:
        return pd.DataFrame()

    page_size = int(max(1, min(int(page_size), MAX_ALLB_PAGE_SIZE)))
    page = int(max(0, page))

    start = page * page_size
    if start >= total:
        start = max(0, (total - 1) // page_size) * page_size
        page = start // page_size

    end = min(start + page_size, total)
    if end <= start:
        return pd.DataFrame()

    # localizar batch inicial/final
    # cum[i] = total até i
    start_b = bisect_right(cum, start)
    end_b = bisect_right(cum, end - 1)

    rows_out = []

    prev_cum_start = cum[start_b - 1] if start_b > 0 else 0
    global_cursor = start

    for bi in range(start_b, end_b + 1):
        if bi < 0 or bi >= len(batch_order):
            break
        bn = batch_order[bi]
        c = int(counts[bi]) if bi < len(counts) else 0
        if c <= 0:
            continue

        prev_cum = cum[bi - 1] if bi > 0 else 0
        batch_global_start = prev_cum
        batch_global_end = prev_cum + c

        seg_start = max(global_cursor, batch_global_start)
        seg_end = min(end, batch_global_end)
        if seg_end <= seg_start:
            continue

        local_start = seg_start - batch_global_start
        local_end = seg_end - batch_global_start

        n = int(lens.get(bn, 0))
        pack = packmasks.get(bn, None)
        if pack is None or n <= 0:
            continue

        mask_bits = np.unpackbits(np.frombuffer(pack, dtype=np.uint8), count=n).astype(bool)
        idx_sel = np.flatnonzero(mask_bits)
        if idx_sel.size == 0:
            continue

        take = idx_sel[int(local_start): int(local_end)]
        if take.size == 0:
            continue

        csv_path = join(str(s_ctx.get("csv_folder", "")), bn)
        if not exists(csv_path):
            continue

        dfb = _read_batch_full_for_page(csv_path, bn)
        if len(dfb) <= 0:
            continue

        take = take[take < len(dfb)]
        if take.size == 0:
            continue

        part = dfb.iloc[take].copy()
        part["csv_basename"] = bn
        part["custom_data_orig"] = part["custom_data"].astype(str)
        part["custom_data"] = [_global_id(bn, cd) for cd in part["custom_data_orig"].astype(str).tolist()]

        # widths/heights (somente para esses itens)
        try:
            img_dir = _images_dir_for_csv(s_ctx, bn)
            w, h = compute_img_ratios(img_dir, part["names"])
            part["widths"] = w
            part["heights"] = h
        except Exception:
            part["widths"] = 1
            part["heights"] = 1

        rows_out.append(part)

        global_cursor = seg_end

    if not rows_out:
        return pd.DataFrame()

    out = pd.concat(rows_out, ignore_index=True)
    # reforça dtypes
    for c in ["names", "custom_data", "manual_label", "correct_label", "csv_basename", "custom_data_orig"]:
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out


# =========================
# CLUSTER (subset-based dendrogram)
# =========================
def _build_dendrogram_figure(Z: np.ndarray, cut_dist: float, hard_cap: int = 20000) -> go.Figure:
    fig = go.Figure()

    if dendrogram is None:
        fig.add_annotation(text="scipy dendrogram não disponível.", showarrow=False)
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    if Z is None or (not hasattr(Z, "size")) or Z.size == 0:
        fig.add_annotation(text="Sem dados suficientes para dendrograma.", showarrow=False)
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    Z = np.asarray(Z, dtype=np.float64)
    n = Z.shape[0] + 1
    if n > hard_cap:
        fig.add_annotation(
            text=f"Dendrograma não desenhado (N={n} > hard_cap={hard_cap}). Reduza a seleção/filtros.",
            showarrow=False,
        )
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    d = dendrogram(Z, no_plot=True, color_threshold=cut_dist)

    icoord = d["icoord"]
    dcoord = d["dcoord"]
    color_list = d.get("color_list", ["C0"] * len(icoord))

    mpl_cycle = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    def _map_mpl_color(c: str) -> str:
        if isinstance(c, str) and len(c) == 2 and c[0] == "C" and c[1].isdigit():
            idx = int(c[1])
            return mpl_cycle[idx % len(mpl_cycle)]
        return c if isinstance(c, str) else "#1f77b4"

    traces = {}
    for xs, ys, col in zip(icoord, dcoord, color_list):
        is_above = (max(ys) >= cut_dist)
        key_color = "black" if is_above else _map_mpl_color(col)
        if key_color not in traces:
            traces[key_color] = {"x": [], "y": []}
        traces[key_color]["x"].extend([xs[0], xs[1], xs[2], xs[3], None])
        traces[key_color]["y"].extend([ys[0], ys[1], ys[2], ys[3], None])

    for c, xy in traces.items():
        fig.add_trace(
            go.Scatter(
                x=xy["x"],
                y=xy["y"],
                mode="lines",
                line=dict(color=c, width=1.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_hline(y=cut_dist, line_dash="dash", line_color="black", opacity=0.9)

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title="distance", showgrid=False, zeroline=False),
    )
    return fig


# =========================
# INIT (compatível com runner.py)
# =========================
def init(argv):
    """
    runner.py chama:
      python app.py <project> [graph_range]
    """
    if len(argv) < 2:
        raise AssertionError("Uso: python app.py <project> [graph_range]")

    project_name = argv[1]
    graph_range = 80
    if len(argv) >= 3:
        graph_range = int(argv[2])

    project_dir = join(ASSETS_FOLDER, project_name)
    if not isdir(project_dir):
        raise AssertionError(f"Projeto não encontrado em: {project_dir}")

    csv_folder = join(project_dir, "dataframes")
    csv_file = _pick_first_csv(csv_folder)
    csv_basename = basename(csv_file)

    stem = csv_basename[:-4]
    batch_folder = _infer_batch_folder_from_csv_basename(stem)

    images_base = join(project_dir, "images")
    if batch_folder:
        path_to_images = _ensure_samples_dir(join(images_base, batch_folder))
    else:
        path_to_images = _ensure_samples_dir(images_base)

    # thumbnails == images
    path_to_thumbnails = path_to_images

    background_path = _bg_or_blank(_resolve_background(project_name, stem))

    user_id = 0
    port = 8025

    print("[INFO] Dash assets_folder:", ASSETS_FOLDER)
    print("Project:", project_name)
    print("imgs:", path_to_images, f"(imgs={_count_images(path_to_images)})")
    print("thumbnails:", path_to_thumbnails)
    print("csv:", csv_file)
    if background_path == BLANK_BG_PATH:
        print("[INFO] Sem background para este batch (usando blank).")

    return (
        path_to_images,
        path_to_thumbnails,
        csv_file,
        csv_folder,
        csv_basename,
        background_path,
        project_name,
        user_id,
        port,
        graph_range,
    )


# =========================
# INIT APP STATE
# =========================
(
    path_to_images,
    path_to_thumbnails,
    csv_file,
    csv_folder,
    csv_basename,
    background_path,
    project_name,
    user_id,
    port,
    graph_range,
) = init(sys.argv)

df = _ensure_required_columns(read_input_csv(csv_file))
BATCH_DF_CACHE[csv_basename] = df.copy()

thumbs_jpg = ("thumbnails" in df.columns)
HAS_VAR = ("var_norm" in df.columns)

# (no batch inicial, ainda calculamos ratios como antes para esse DF)
try:
    widths, heights = compute_img_ratios(path_to_images, df["names"])
    df["widths"] = widths
    df["heights"] = heights
except Exception:
    df["widths"] = 1
    df["heights"] = 1

background_ranges = {"x": [-graph_range, graph_range], "y": [-graph_range, graph_range]}
background_color = init_appearance()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
    assets_folder=ASSETS_FOLDER,
    assets_url_path=ASSETS_URL_PATH,
)

labels_colors, next_label_id = _init_labels_colors(df)

map_columns_vals = {
    "area": "Seg Area",
    "perimeter": "Perimeter",
    "circularity": "Circularity",
    "avg_contour": "Avg Contour",
    "avg_intensity": "Avg Intensity",
    "elongation": "Elongation",
}
dropdown_image_vals = _compute_dropdown_image_vals(df, map_columns_vals)
csv_candidates = _list_csv_candidates(csv_folder)

store_context_init = dict(
    project_name=project_name,
    csv_folder=csv_folder,
    csv_basename=csv_basename,             # batch atual real (mesmo se iniciar em overview)
    csv_file=csv_file,
    path_to_images=path_to_images,         # FS path
    path_to_thumbnails=path_to_images,     # FS path (igual)
    background_path=background_path,       # FS path (blank-safe)
    thumbs_jpg=thumbs_jpg,
    has_var=HAS_VAR,
    images_path_template=path_to_images,   # FS template (contém batch_####/samples)
    is_overview=True,                      # abre já no overview
    overview_version=0,                    # incrementa a cada relabel/undo/cluster relabel
    allb_active=False,                     # all-batches selection ativo?
)

# =========================
# UI COMPONENTS
# =========================
header = html.H1(
    "Image Labeling Tool - OVERVIEW",
    id="header_title",
    style={"color": "CornflowerBlue", "marginBottom": "0.5rem"},
)

batch_dropdown = dcc.Dropdown(
    options=[{"label": "Overview (all batches)", "value": OVERVIEW_VALUE}]
            + [{"label": f, "value": f} for f in csv_candidates],
    value=OVERVIEW_VALUE,
    id="dropdown_batch",
    clearable=False,
    style={"width": "100%"},
)

# --- all-batches UI ---
allb_card = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                "Multi-Batch Selection",
                style={
                    "fontWeight": "600",
                    "marginBottom": "10px",
                    "fontSize": "14px",
                    "color": "#333"
                }
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            "Classes: mesmo filtro da legenda do plot",
                            style={
                                "fontSize": "12px", 
                                "marginBottom": "2px",
                                "color": "#666",
                                "fontStyle": "italic"
                            }
                        ),
                        width=12,
                    ),
                ]
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        html.Div("Page Size", style={"fontSize": "12px", "marginBottom": "2px"}),
                        width=12,
                    ),
                ]
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Input(
                            id="input_allb_page_size",
                            type="number",
                            min=100,
                            max=MAX_ALLB_PAGE_SIZE,
                            step=100,
                            value=2000,
                            style={"width": "100%", "fontSize": "12px", "padding": "6px"},
                        ),
                        width=12,
                    ),
                ],
                align="center",
                className="mb-2",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Build Selection",
                            id="button_allb_build",
                            n_clicks=0,
                            color="primary",
                            size="sm",
                            style={"width": "100%"},
                        ),
                        width=12,
                    ),
                ],
                className="mb-2",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.ButtonGroup(
                            [
                                dbc.Button("◀ Prev", id="button_allb_prev", n_clicks=0, 
                                          color="secondary", size="sm"),
                                dbc.Button("Next ▶", id="button_allb_next", n_clicks=0, 
                                          color="secondary", size="sm"),
                            ],
                            style={"width": "100%"},
                        ),
                        width=8,
                    ),
                    dbc.Col(
                        html.Div(
                            id="allb_page_info", 
                            style={
                                "fontFamily": "monospace", 
                                "fontSize": "11px",
                                "padding": "5px",
                                "backgroundColor": "#f8f9fa",
                                "borderRadius": "3px",
                                "border": "1px solid #dee2e6"
                            }
                        ),
                        width=4,
                    ),
                ],
                align="center",
            ),
        ]
    ),
    className="mb-3",
)

# --- Labeling Controls ---
labeling_card = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                "Labeling",
                style={
                    "fontWeight": "600",
                    "marginBottom": "10px",
                    "fontSize": "14px",
                    "color": "#333"
                }
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            placeholder="Enter new label",
                            id="input_aplicar_novo_label",
                            type="text",
                            style={"fontSize": "12px"},
                        ),
                        width=8,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Apply",
                            id="button_aplicar_novo_label",
                            color="primary",
                            size="sm",
                            style={"width": "100%"},
                        ),
                        width=4,
                    ),
                ],
                className="mb-2",
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checklist(
                            options=[
                                {"label": " Save after labeling", "value": " Save CSV after labeling"}
                            ],
                            value=[" Save CSV after labeling"],
                            id="check_save_csv",
                            inline=True,
                            style={"fontSize": "12px"},
                        ),
                        width=12,
                    ),
                ],
                className="mb-2",
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checklist(
                            options=[
                                {"label": " Discard after labeling", "value": " Discard images after labeling"}
                            ],
                            value=[" Discard images after labeling"],
                            id="check_discard_marked",
                            inline=True,
                            style={"fontSize": "12px"},
                        ),
                        width=12,
                    ),
                ],
            ),
        ]
    ),
    className="mb-3",
)

# --- Image Display Controls ---
display_card = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                "Display Options",
                style={
                    "fontWeight": "600",
                    "marginBottom": "10px",
                    "fontSize": "14px",
                    "color": "#333"
                }
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        html.Div("Marker Size", style={"fontSize": "12px", "marginBottom": "5px"}),
                        width=12,
                    ),
                    dbc.Col(
                        dcc.Slider(
                            1, 25, 3,
                            value=10,
                            id="slider_marker_size",
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        width=12,
                    ),
                ],
                className="mb-2",
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        html.Div("Label Order", style={"fontSize": "12px", "marginBottom": "2px"}),
                        width=12,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            ["A-Z, a-z", "Frequency"],
                            value="A-Z, a-z",
                            id="dropdown_order_labels",
                            clearable=False,
                            style={"fontSize": "12px"},
                        ),
                        width=12,
                    ),
                ],
                className="mb-3",
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        html.Div("Image Order", style={"fontSize": "12px", "marginBottom": "2px"}),
                        width=12,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            dropdown_image_vals,
                            value=["Similarity"],
                            id="dropdown_order_images",
                            clearable=False,
                            multi=True,
                            style={"fontSize": "12px"},
                        ),
                        width=12,
                    ),
                ],
                className="mb-2",
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checklist(
                            options=[
                                {"label": " Marked first", "value": " Marked images first"}
                            ],
                            value=[],
                            id="check_marked_first",
                            inline=True,
                            style={"fontSize": "12px"},
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Checklist(
                            options=[
                                {"label": " Hide relabeled", "value": " Hide relabeled images"}
                            ],
                            value=[],
                            id="check_discard_relabeled",
                            inline=True,
                            style={"fontSize": "12px"},
                        ),
                        width=6,
                    ),
                ],
                className="mb-3",
            ),
        ]
    ),
    className="mb-3",
)

# --- Filtering Controls ---
filtering_card = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                "Filtering",
                style={
                    "fontWeight": "600",
                    "marginBottom": "10px",
                    "fontSize": "14px",
                    "color": "#333"
                }
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                "Confidence Range", 
                                style={"fontSize": "12px", "marginBottom": "5px", "textAlign": "center"}
                            ),
                            dcc.RangeSlider(
                                0, 1, 0.05,
                                value=[0, 1],
                                id="probs_range_slider",
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        width=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                "Variance Range", 
                                style={"fontSize": "12px", "marginBottom": "5px", "textAlign": "center"}
                            ),
                            dcc.RangeSlider(
                                0, 1, 0.05,
                                value=[0, 1],
                                id="vars_range_slider",
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        width=6,
                        id="col_vars_slider",
                        style={"display": "block" if HAS_VAR else "none"},
                        className="mb-3",
                    ),
                ]
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Invert Selection",
                            id="button_invert_marks",
                            n_clicks=0,
                            color="secondary",
                            size="sm",
                            style={"width": "100%"},
                        ),
                        width=12,
                    ),
                ],
            ),
        ]
    ),
    className="mb-3",
)

# --- Cluster UI ---
# --- Cluster UI ---
cluster_card = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                "Clustering",
                style={
                    "fontWeight": "600", 
                    "marginBottom": "10px",
                    "fontSize": "14px",
                    "color": "#333"
                }
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        html.Div("Target Class", style={"fontSize": "12px", "marginBottom": "2px"}),
                        width=12,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            options=[{"label": x, "value": x} for x in sorted(df["manual_label"].astype(str).unique().tolist())],
                            value=None,
                            id="dropdown_cluster_class",
                            placeholder="Select class for clustering",
                            clearable=True,
                            style={"fontSize": "12px"},
                        ),
                        width=12,
                        className="mb-2",
                    ),
                ]
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        html.Div("Distance Cut", style={"fontSize": "12px", "marginBottom": "5px"}),
                        width=12,
                    ),
                    dbc.Col(
                        dcc.Slider(
                            min=0.0,
                            max=1.0,
                            step=0.01,
                            value=0.50,
                            id="slider_cluster_cut",
                            marks={
                                0: {'label': '0', 'style': {'font-size': '12px'}},
                                0.5: {'label': '0.5', 'style': {'font-size': '12px'}},
                                1.0: {'label': '1', 'style': {'font-size': '12px'}}
                            },
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                                "template": "Cut: {value:.2f}"
                            },
                        ),
                        width=12,
                        className="mb-3",
                    ),
                ]
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Update Cluster",
                            id="button_update_cluster",
                            n_clicks=0,
                            color="primary",
                            size="sm",
                            style={"width": "100%"},
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Apply Cluster Labels",
                            id="button_relabel_cluster",
                            n_clicks=0,
                            color="success",
                            size="sm",
                            style={"width": "100%"},
                        ),
                        width=6,
                    ),
                ],
                className="mb-3",
            ),
            
            html.Div(
                id="cluster_summary",
                style={
                    "marginTop": "8px", 
                    "fontFamily": "monospace", 
                    "whiteSpace": "pre-wrap", 
                    "fontSize": "11px",
                    "padding": "8px",
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "3px",
                    "border": "1px solid #dee2e6"
                }
            ),
            
            dcc.Graph(
                id="graph_cluster_dendrogram",
                figure=go.Figure(),
                config={"displayModeBar": False},
                style={"marginTop": "10px", "height": "220px"},
            ),
        ]
    ),
    className="mb-3",
)

# --- Save Controls ---
save_card = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                "Save & Export",
                style={
                    "fontWeight": "600",
                    "marginBottom": "10px", 
                    "fontSize": "14px",
                    "color": "#333"
                }
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        html.Div("CSV Name", style={"fontSize": "12px", "marginBottom": "2px"}),
                        width=12,
                    ),
                    dbc.Col(
                        dbc.Input(
                            value=csv_basename,
                            id="input_save_csv",
                            type="text",
                            style={"fontSize": "12px"},
                        ),
                        width=12,
                        className="mb-2",
                    ),
                ]
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Save CSV",
                            id="button_save_csv",
                            n_clicks=0,
                            color="primary",
                            size="sm",
                            style={"width": "100%"},
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "UNDO",
                            id="button_undo",
                            n_clicks=0,
                            color="warning",
                            size="sm",
                            style={"width": "100%"},
                        ),
                        width=6,
                    ),
                ],
                className="mb-3",
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        html.Div("Dataset Name", style={"fontSize": "12px", "marginBottom": "2px"}),
                        width=12,
                    ),
                    dbc.Col(
                        dbc.Input(
                            value="annotated_dataset",
                            id="input_finish_work",
                            type="text",
                            style={"fontSize": "12px"},
                        ),
                        width=12,
                        className="mb-2",
                    ),
                ]
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Export Dataset",
                            id="button_finish_work",
                            n_clicks=0,
                            color="success",
                            size="sm",
                            style={"width": "100%"},
                        ),
                        width=12,
                    ),
                ]
            ),
        ]
    ),
    className="mb-3",
)

IMAGES = create_list_dics(_list_src=[], _list_thumbnail=[], _list_name_figure=[])

# figura inicial: OVERVIEW (não batch)
fig_init = _get_overview_fig_cached(project_name, csv_folder, version=0, graph_range=graph_range)

# =========================
# LAYOUT IMPROVEMENTS
# =========================
app.layout = html.Div(
    [
        dbc.Container(
            fluid=True,
            children=[
                # Header Row
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                header,
                                style={"textAlign": "left"}
                            ),
                            width=8,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div("Select Batch", style={"fontSize": "12px", "color": "#666", "marginBottom": "4px"}),
                                    batch_dropdown,
                                ],
                                style={"paddingTop": "15px"}
                            ),
                            width=4,
                        ),
                    ],
                    align="center",
                    className="mb-4 mt-3",
                ),
                
                # Main Content Row
                dbc.Row(
                    [
                        # Left Column - Scatter Plot
                        dbc.Col(
                            [
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="g_scatter_plot",
                                                figure=fig_init,
                                                style={"height": "74vh", "minHeight": "500px"},
                                                config={
                                                    "displaylogo": False,
                                                    "modeBarButtonsToRemove": ["toImage", "resetScale2d"],
                                                    "displayModeBar": True,
                                                },
                                            ),
                                        ],
                                        style={"padding": "10px"}
                                    ),
                                    style={"height": "100%", "border": "1px solid #dee2e6"}
                                ),
                            ],
                            width=7,
                            className="mb-4",
                        ),
                        
                        # Right Column - Controls
                        dbc.Col(
                            [
                                # Scrollable controls container
                                html.Div(
                                    [
                                        allb_card,
                                        labeling_card,
                                        display_card,
                                        filtering_card,
                                        cluster_card,
                                        save_card,
                                    ],
                                    style={
                                        "height": "74vh",
                                        "minHeight": "500px",
                                        "overflowY": "auto",
                                        "paddingRight": "10px"
                                    }
                                ),
                            ],
                            width=5,
                            className="mb-4",
                        ),
                    ],
                    className="mb-4",
                ),
                
                # Image Gallery Row
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                "Image Gallery",
                                                style={
                                                    "fontWeight": "600",
                                                    "marginBottom": "10px",
                                                    "fontSize": "14px",
                                                    "color": "#333"
                                                }
                                            ),
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
                                                style={
                                                    "height": "44vh",
                                                    "minHeight": "300px",
                                                    "overflow": "auto",
                                                    "backgroundColor": background_color,
                                                    "padding": "10px",
                                                    "borderRadius": "5px",
                                                    "border": "1px solid #dee2e6"
                                                }
                                            ),
                                        ]
                                    ),
                                    style={"border": "1px solid #dee2e6"}
                                ),
                            ],
                            width=12,
                        ),
                    ],
                    className="mb-4",
                ),
                
                # Alerts Row
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Alert(
                                    id="alert_allb",
                                    children="",
                                    is_open=False,
                                    color="info",
                                    dismissable=True,
                                    duration=6000,
                                ),
                                dbc.Alert(
                                    id="alert_cluster_problem",
                                    children="",
                                    is_open=False,
                                    color="warning",
                                    dismissable=True,
                                    duration=6000,
                                ),
                            ],
                            width=12,
                            className="mb-3",
                        ),
                    ]
                ),
            ]
        ),
        
        # Hidden Stores
        dcc.Store(id="selected_custom_points", data=json.dumps([])),
        dcc.Store(id="unchecked_points", data=json.dumps([])),
        dcc.Store(id="overview_hidden_groups", data=json.dumps([])),
        dcc.Store(id="state_store_df", data=df.to_json()),
        dcc.Store(id="store_context", data=store_context_init),
        dcc.Store(id="cluster_meta_store", data=json.dumps({"ready": False})),
        dcc.Store(id="allb_store", data=json.dumps({"active": False, "token": None, "page": 0, "page_size": 2000, "total": 0})),
        dcc.Store(id="dummy_csv_save", data=0),
        dcc.Store(id="dummy_finish_work", data=0),
        html.Div(id="output", style={"display": "none"}),
    ],
    style={
        "backgroundColor": "#f8f9fa",
        "minHeight": "100vh",
        "padding": "20px"
    }
)

@app.callback(
    Output("overview_hidden_groups", "data"),
    Input("g_scatter_plot", "restyleData"),
    State("g_scatter_plot", "figure"),
    State("overview_hidden_groups", "data"),
    prevent_initial_call=True,
)
def _sync_overview_hidden_groups(restyleData, fig, hidden_json):
    try:
        hidden = set(json.loads(hidden_json) if hidden_json else [])
    except Exception:
        hidden = set()

    if not restyleData or not fig or "data" not in fig:
        return json.dumps(sorted(hidden))

    edits, idxs = restyleData
    if not isinstance(idxs, list):
        idxs = [idxs]

    vis = edits.get("visible") if isinstance(edits, dict) else None

    for j, idx in enumerate(idxs):
        try:
            tr = fig["data"][idx]
        except Exception:
            continue

        group = tr.get("legendgroup") or tr.get("name")
        if not group:
            continue

        vj = vis[j] if isinstance(vis, list) and j < len(vis) else vis

        if vj == "legendonly" or vj is False:
            hidden.add(group)
        elif vj is True:
            hidden.discard(group)

    return json.dumps(sorted(hidden))
    
# =========================
# HELPERS (mantidos da versão original)
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


def _ensure_color_for_label(lab: str):
    global labels_colors, next_label_id
    if lab not in labels_colors:
        labels_colors[lab] = next_label_id
        next_label_id += 1
    return labels_colors[lab]


def _format_allb_page_info(allb: dict) -> str:
    if not allb or not allb.get("active"):
        return "inactive"
    total = int(allb.get("total", 0))
    page = int(allb.get("page", 0))
    page_size = int(allb.get("page_size", 2000))
    pages = (total + page_size - 1) // page_size if page_size > 0 else 0
    return f"page={page+1}/{max(pages,1)} | page_size={page_size} | total={total}"


# =========================
# CALLBACKS (mantidos da versão original)
# =========================
@app.callback(
    [Output("dummy_finish_work", "data")],
    [Input("button_finish_work", "n_clicks")],
    [State("state_store_df", "data"), State("input_finish_work", "value"), State("store_context", "data")],
)
def save_dataset(i_button_save_csv_nclicks, s_store_df, s_input_finish_work_value, s_ctx):
    if i_button_save_csv_nclicks > 0:
        s_ctx = s_ctx or {}
        is_overview = s_ctx.get("is_overview", False)
        
        # CORREÇÃO: Obter csv_folder do contexto OU usar a variável global
        project_csv_folder = s_ctx.get("csv_folder")
        if project_csv_folder is None:
            # Usar a variável global csv_folder (definida no início do script)
            project_csv_folder = globals().get('csv_folder', '')

        out_root = str(s_input_finish_work_value)
        if not exists(out_root):
            mkdir(out_root)

        # MODIFICAÇÃO: Se estiver no modo overview, exportar todos os batches
        if is_overview:
            print(f"[INFO] Modo OVERVIEW ativo. Exportando TODOS os batches...")
            print(f"[INFO] CSV Folder: {project_csv_folder}")
            
            if not project_csv_folder or not exists(project_csv_folder):
                print(f"[ERRO] Pasta de CSVs não encontrada: {project_csv_folder}")
                return [0]
            
            # Listar todos os CSVs (exceto o valor especial OVERVIEW)
            csv_candidates = _list_csv_candidates(project_csv_folder)
            csv_files = [f for f in csv_candidates if f != OVERVIEW_VALUE]
            
            if not csv_files:
                print("[ERRO] Nenhum arquivo CSV encontrado na pasta!")
                return [0]
                
            print(f"[INFO] {len(csv_files)} batches encontrados para exportação")
            
            total_images = 0
            total_classes = set()
            
            for csv_basename in csv_files:
                try:
                    csv_path = join(project_csv_folder, csv_basename)
                    
                    if not exists(csv_path):
                        print(f"[WARN] CSV não encontrado: {csv_path}")
                        continue
                    
                    # Carregar o batch atual (usando cache se disponível)
                    if csv_basename in BATCH_DF_CACHE:
                        df_batch = BATCH_DF_CACHE[csv_basename].copy()
                        print(f"  - Batch {csv_basename}: usando cache ({len(df_batch)} registros)")
                    else:
                        df_batch = _ensure_required_columns(read_input_csv(csv_path))
                        print(f"  - Batch {csv_basename}: lendo do disco ({len(df_batch)} registros)")
                    
                    # Determinar diretório de imagens para este batch
                    img_dir = _images_dir_for_csv(s_ctx, csv_basename)
                    
                    if not exists(img_dir):
                        print(f"[WARN] Diretório de imagens não encontrado: {img_dir}")
                        continue
                    
                    # Para cada linha do batch, copiar a imagem para a pasta do label
                    batch_images = 0
                    for _, row in df_batch.iterrows():
                        try:
                            label = str(row["manual_label"])
                            name = str(row["names"])
                            
                            # Criar diretório para o label se não existir
                            dst_dir = join(out_root, label)
                            if not exists(dst_dir):
                                mkdir(dst_dir)
                                total_classes.add(label)
                            
                            # Caminhos de origem e destino
                            src = join(img_dir, name)
                            dst = join(dst_dir, name)
                            
                            if not exists(src):
                                print(f"[WARN] Imagem não encontrada: {src}")
                                continue
                            
                            # Copiar imagem (evitar sobrescrever)
                            if not exists(dst):
                                shutil.copy2(src, dst)
                                total_images += 1
                                batch_images += 1
                                
                        except Exception as e:
                            print(f"[WARN] Falha ao copiar {name} do batch {csv_basename}: {e}")
                    
                    print(f"    ✓ {batch_images} imagens exportadas")
                
                except Exception as e:
                    print(f"[ERRO] Erro ao processar batch {csv_basename}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"\n[SUCESSO] Dataset exportado no modo OVERVIEW!")
            print(f"  - Batches processados: {len(csv_files)}")
            print(f"  - Total de imagens exportadas: {total_images}")
            print(f"  - Classes criadas: {len(total_classes)}")
            print(f"  - Diretório de saída: {out_root}\n")
            
        else:
            # Código original para modo batch normal
            df_updated = pd.read_json(StringIO(s_store_df))
            
            # se estiver em allb, pode ter múltiplos batches na página
            if "csv_basename" in df_updated.columns:
                # cria pastas
                for label in df_updated["manual_label"].astype(str).unique().tolist():
                    out_dir = join(out_root, label)
                    if not exists(out_dir):
                        mkdir(out_dir)

                for _, row in df_updated.iterrows():
                    try:
                        bn = str(row.get("csv_basename", ""))
                        name = str(row["names"])
                        folder = str(row["manual_label"])
                        src_dir = _images_dir_for_csv(s_ctx, bn) if bn else str(s_ctx.get("path_to_images", ""))
                        src = join(src_dir, name)
                        dst = join(out_root, folder, name)
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"[WARN] copy failed: {e}")
            else:
                # modo batch normal
                for label in df_updated["manual_label"].astype(str).unique().tolist():
                    out_dir = join(out_root, label)
                    if not exists(out_dir):
                        mkdir(out_dir)

                path_imgs = s_ctx.get("path_to_images", "")
                for _, row in df_updated.iterrows():
                    name = str(row["names"])
                    folder = str(row["manual_label"])
                    src = join(path_imgs, name)
                    dst = join(out_root, folder, name)
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"[WARN] copy failed: {src} -> {dst} ({e})")

            print("\nDataset recorded.\n")

    return [0]

@app.callback(
    [Output("dummy_csv_save", "data")],
    [Input("button_save_csv", "n_clicks")],
    [State("state_store_df", "data"), State("input_save_csv", "value"), State("store_context", "data")],
)
def button_save_csv(i_button_save_csv_nclicks, s_store_df, s_input_save_csv_value, s_ctx):
    """
    - batch normal: salva o CSV atual
    - allb ativo: salva somente batches dirty (em memória) com seus nomes originais
    - overview: salva TODOS os batches que foram modificados
    """
    if i_button_save_csv_nclicks > 0:
        s_ctx = s_ctx or {}
        allb_active = bool(s_ctx.get("allb_active", False))
        is_overview = bool(s_ctx.get("is_overview", False))

        if is_overview and not allb_active:
            # Modo overview: salvar todos os batches que foram modificados
            print(f"[INFO] Modo OVERVIEW ativo. Salvando TODOS os batches modificados...")
            
            if not DIRTY_BATCHES:
                print("[INFO] Nenhum batch modificado para salvar.")
                return [0]
            
            # Obter a pasta de CSVs
            project_csv_folder = s_ctx.get("csv_folder")
            if project_csv_folder is None:
                project_csv_folder = globals().get('csv_folder', '')
            
            saved_count = 0
            for bn in sorted(DIRTY_BATCHES):
                try:
                    if bn not in BATCH_DF_CACHE:
                        print(f"[WARN] Batch {bn} no cache.")
                        continue
                    
                    csv_path = join(project_csv_folder, bn)
                    if not exists(csv_path):
                        print(f"[WARN] Caminho CSV não existe: {csv_path}")
                        continue
                    
                    save_csv(BATCH_DF_CACHE[bn], bn)
                    saved_count += 1
                    print(f"  ✓ Batch salvo: {bn}")
                except Exception as e:
                    print(f"[ERRO] Falha ao salvar batch {bn}: {e}")
            
            print(f"[SUCESSO] {saved_count} batches salvos.\n")
            
            # Limpar dirty batches após salvar
            DIRTY_BATCHES.clear()
            return [0]

        if allb_active:
            # Obter a pasta de CSVs
            project_csv_folder = s_ctx.get("csv_folder")
            if project_csv_folder is None:
                project_csv_folder = globals().get('csv_folder', '')
            
            # salva somente os dirty batches (evita acidente de sobrescrever tudo)
            for bn in sorted(DIRTY_BATCHES):
                try:
                    if bn not in BATCH_DF_CACHE:
                        continue
                    save_csv(BATCH_DF_CACHE[bn], bn)
                except Exception as e:
                    print(f"[WARN] save dirty batch failed: {bn}: {e}")
            return [0]

        df_updated = pd.read_json(StringIO(s_store_df))
        if "custom_data" in df_updated.columns:
            df_updated["custom_data"] = df_updated["custom_data"].astype(str)
        if "names" in df_updated.columns:
            df_updated["names"] = df_updated["names"].astype(str)

        # atualiza cache do batch atual (mantém coerência com overview)
        csv_bn = str(s_ctx.get("csv_basename", ""))
        if csv_bn and csv_bn != OVERVIEW_VALUE:
            BATCH_DF_CACHE[csv_bn] = df_updated.copy()

        save_csv(df_updated, str(s_input_save_csv_value))
    return [0]

@app.callback(
    [
        Output("selected_custom_points", "data"),
        Output("state_store_df", "data"),
        Output("unchecked_points", "data"),
        Output("store_context", "data"),
        Output("header_title", "children"),
        Output("input_save_csv", "value"),
        Output("dropdown_order_images", "options"),
        Output("dropdown_order_images", "value"),
        Output("col_vars_slider", "style"),
        Output("allb_store", "data"),
        Output("allb_page_info", "children"),
        Output("alert_allb", "children"),
        Output("alert_allb", "is_open"),
        Output("output", "children"),
    ],
    [
        Input("g_scatter_plot", "selectedData"),
        Input("button_allb_build", "n_clicks"),
        Input("button_allb_prev", "n_clicks"),
        Input("button_allb_next", "n_clicks"),
        Input("button_aplicar_novo_label", "n_clicks"),
        Input("button_relabel_cluster", "n_clicks"),
        Input("button_invert_marks", "n_clicks"),
        Input("button_undo", "n_clicks"),
        Input("dropdown_batch", "value"),
    ],
    [
        State("selected_custom_points", "data"),
        State("unchecked_points", "data"),
        State("state_store_df", "data"),
        State("g_image_selector", "images"),
        State("input_aplicar_novo_label", "value"),
        State("check_save_csv", "value"),
        State("input_save_csv", "value"),
        State("check_discard_marked", "value"),
        State("store_context", "data"),
        State("cluster_meta_store", "data"),
        State("input_allb_page_size", "value"),
        State("probs_range_slider", "value"),
        State("vars_range_slider", "value"),
        State("check_discard_relabeled", "value"),
        State("allb_store", "data"),
        State("g_scatter_plot", "figure"),
        State("overview_hidden_groups", "data"),
    ],
)
def mudanca_custom_data(
    i_selection_g_scatter_plot,
    i_allb_build_clicks,
    i_allb_prev_clicks,
    i_allb_next_clicks,
    i_button_aplicar_novo_label_nclicks,
    i_button_relabel_cluster_nclicks,
    i_invert_marks_nclicks,
    i_button_undo_nclicks,
    i_dropdown_batch_value,
    s_selected_custom_points,
    s_unchecked_points,
    s_store_df,
    s_button_galeria_filtrar_images,
    s_input_aplicar_novo_label_value,
    s_check_save_csv_value,
    s_input_save_csv_value,
    s_check_discard_marked_value,
    s_ctx,
    s_cluster_meta_json,
    s_allb_page_size,
    s_probs_rng,
    s_vars_rng,
    s_hide_relab_val,
    s_allb_store_json,
    s_g_scatter_plot_figure,
    s_overview_hidden_groups_json,
):
    global labels_colors, next_label_id, csv_folder, UNDO_SNAPSHOT

    out_ctx = dict(s_ctx) if isinstance(s_ctx, dict) else {}
    out_ctx.setdefault("overview_version", 0)
    out_ctx.setdefault("is_overview", False)
    out_ctx.setdefault("allb_active", False)

    try:
        allb_store = json.loads(s_allb_store_json) if s_allb_store_json else {"active": False}
    except Exception:
        allb_store = {"active": False}

    # lê DF atual (pode ser batch normal ou "page DF" do allb)
    df_updated = pd.read_json(StringIO(s_store_df))
    for c in ["custom_data", "names", "manual_label", "correct_label"]:
        if c in df_updated.columns:
            df_updated[c] = df_updated[c].astype(str)

    ctx = dash.callback_context
    flag_callback = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

    df_store_updated = df_updated.to_json()
    selectedpoints = s_selected_custom_points if s_selected_custom_points else json.dumps([])
    unchecked_points = []
    out_header = dash.no_update
    out_input_save_csv = dash.no_update
    out_dropdown_opts = dash.no_update
    out_dropdown_val = dash.no_update
    out_vars_style = dash.no_update
    out_alert = ""
    out_alert_open = False
    out_msg = ""

    def _invalidate_overview():
        if OVERVIEW_CACHE.get("project") == out_ctx.get("project_name"):
            OVERVIEW_CACHE["version"] = None
            OVERVIEW_CACHE["fig"] = None

    def _bump_overview_version():
        out_ctx["overview_version"] = int(out_ctx.get("overview_version", 0)) + 1
        _invalidate_overview()

    def _current_allb_token() -> str | None:
        if allb_store and allb_store.get("active") and allb_store.get("token"):
            return str(allb_store.get("token"))
        return None

    def _refresh_allb_after_batch_changes(changed_batches: list[str]):
        """
        Recalcula máscaras para batches alterados e atualiza:
          - allb_store.total
          - df_store_updated para página atual (reclamp)
        """
        nonlocal df_updated, df_store_updated, allb_store, selectedpoints, s_unchecked_points, out_alert, out_alert_open

        token = _current_allb_token()
        if not token:
            return

        _allb_recompute_batches(token, str(out_ctx.get("csv_folder", "")), changed_batches)
        total = _allb_total(token)

        # reclampa página
        page_size = int(allb_store.get("page_size", 2000))
        page = int(allb_store.get("page", 0))
        pages = (total + page_size - 1) // page_size if page_size > 0 else 1
        if page >= pages:
            page = max(0, pages - 1)

        allb_store["total"] = int(total)
        allb_store["page"] = int(page)

        # recarrega page df
        if total > 0:
            df_page = _allb_make_page_df(token, page, page_size, out_ctx)
        else:
            df_page = pd.DataFrame()

        df_updated = df_page.copy()
        df_store_updated = df_updated.to_json()
        selectedpoints = json.dumps([])
        s_unchecked_points = json.dumps([])

        out_alert = f"[allb] selection atualizado: total={total} | { _format_allb_page_info(allb_store) }"
        out_alert_open = True

    # =========================
    # Troca batch via UI (inclui OVERVIEW)
    # =========================
    if flag_callback == "dropdown_batch":
        chosen = str(i_dropdown_batch_value)

        if chosen == OVERVIEW_VALUE:
            out_ctx["is_overview"] = True
            out_header = "Image Labeling Tool - OVERVIEW"
            selectedpoints = json.dumps([])

            # se já existia allb_store ativo, reconstroi a página atual no store_df
            if allb_store.get("active") and allb_store.get("token"):
                out_ctx["allb_active"] = True
                token = str(allb_store.get("token"))
                total = _allb_total(token)
                allb_store["total"] = int(total)
                page_size = int(allb_store.get("page_size", 2000))
                page = int(allb_store.get("page", 0))
                df_page = _allb_make_page_df(token, page, page_size, out_ctx) if total > 0 else pd.DataFrame()
                df_store_updated = df_page.to_json()
                out_alert = "[OK] Overview ativado com All-batches selection ativo."
                out_alert_open = True
            else:
                out_ctx["allb_active"] = False
                out_alert = "[OK] Overview ativado (heat hull). Use lasso/box + 'Build selection' para All-batches."
                out_alert_open = True

            return [
                selectedpoints,
                df_store_updated,
                json.dumps([]),
                out_ctx,
                out_header,
                out_input_save_csv,
                out_dropdown_opts,
                out_dropdown_val,
                out_vars_style,
                json.dumps(allb_store),
                _format_allb_page_info(allb_store),
                out_alert,
                out_alert_open,
                out_msg,
            ]

        # batch normal: desativa allb (mantém token na memória, mas não ativo na UI)
        out_ctx["is_overview"] = False
        out_ctx["allb_active"] = False
        allb_store["active"] = False

        chosen_csv = chosen
        _csv_folder = out_ctx.get("csv_folder", dirname(out_ctx.get("csv_file", "")))
        csv_folder = _csv_folder
        csv_path = join(_csv_folder, chosen_csv)

        if not exists(csv_path):
            out_msg = f"[ERRO] CSV não encontrado: {csv_path}"
            return [
                json.dumps([]),
                df_store_updated,
                json.dumps([]),
                out_ctx,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                json.dumps(allb_store),
                _format_allb_page_info(allb_store),
                "",
                False,
                out_msg,
            ]

        new_df = _load_batch_df_cached(csv_path, chosen_csv)

        stem = chosen_csv[:-4]
        batch_folder = _infer_batch_folder_from_csv_basename(stem)

        images_template = out_ctx.get("images_path_template", out_ctx.get("path_to_images", ""))

        if batch_folder:
            new_path_imgs = _replace_batch_in_path(images_template, batch_folder)
        else:
            new_path_imgs = out_ctx.get("path_to_images", images_template)

        new_path_imgs = _ensure_samples_dir(new_path_imgs)
        new_path_thumbs = new_path_imgs

        # ratios apenas para este batch inteiro (como antes)
        try:
            w, h = compute_img_ratios(new_path_imgs, new_df["names"])
            new_df = new_df.copy()
            new_df["widths"] = w
            new_df["heights"] = h
        except Exception:
            new_df = new_df.copy()
            new_df["widths"] = 1
            new_df["heights"] = 1

        proj = out_ctx.get("project_name", project_name)
        new_bg = _bg_or_blank(_resolve_background(proj, stem))

        new_thumbs_jpg = ("thumbnails" in new_df.columns)
        new_has_var = ("var_norm" in new_df.columns)

        labels_colors, next_label_id = _init_labels_colors(new_df)

        new_dropdown_opts = _compute_dropdown_image_vals(new_df, map_columns_vals)

        out_ctx.update(
            csv_basename=chosen_csv,
            csv_file=csv_path,
            path_to_images=new_path_imgs,
            path_to_thumbnails=new_path_thumbs,
            background_path=new_bg,
            thumbs_jpg=new_thumbs_jpg,
            has_var=new_has_var,
            images_path_template=images_template,
        )

        df_store_updated = new_df.to_json()
        selectedpoints = json.dumps([])
        s_unchecked_points = json.dumps([])

        out_header = "Image Labeling Tool - " + stem
        out_input_save_csv = chosen_csv
        out_dropdown_opts = new_dropdown_opts
        out_dropdown_val = ["Similarity"]
        out_vars_style = {"display": "block"} if new_has_var else {"display": "none"}

        if len(new_df) > 0:
            example_url = _abs_to_assets_url(join(new_path_imgs, str(new_df["names"].iloc[0])))
            out_msg = f"[OK] Batch carregado: {chosen_csv} | imgs_dir={new_path_imgs} (imgs={_count_images(new_path_imgs)}) | URL_exemplo={example_url}"
        else:
            out_msg = f"[OK] Batch carregado: {chosen_csv} (CSV vazio?)"

        return [
            selectedpoints,
            df_store_updated,
            s_unchecked_points,
            out_ctx,
            out_header,
            out_input_save_csv,
            out_dropdown_opts,
            out_dropdown_val,
            out_vars_style,
            json.dumps(allb_store),
            _format_allb_page_info(allb_store),
            "",
            False,
            out_msg,
        ]

    # =========================
    # Seleção no scatter (NÃO mexe seleção de batch quando em overview/allb)
    # =========================
    if flag_callback == "g_scatter_plot":
        if bool(out_ctx.get("is_overview", False)):
            # não altera selected_custom_points (a seleção no overview serve só para o Build all-batches)
            return [
                selectedpoints,
                df_store_updated,
                s_unchecked_points,
                out_ctx,
                out_header,
                out_input_save_csv,
                out_dropdown_opts,
                out_dropdown_val,
                out_vars_style,
                json.dumps(allb_store),
                _format_allb_page_info(allb_store),
                "",
                False,
                out_msg,
            ]

        if i_selection_g_scatter_plot and i_selection_g_scatter_plot.get("points"):
            selected = [d.get("customdata") for d in i_selection_g_scatter_plot["points"]]
            selected = [str(x) for x in selected if x is not None]
            selectedpoints = json.dumps(selected)
        else:
            selectedpoints = json.dumps([])

    # =========================
    # Build all-batches selection (a partir do overview)
    # =========================    
    elif flag_callback == "button_allb_build":
        out_ctx["is_overview"] = True
        out_ctx["allb_active"] = True
        
        # Usar a seleção filtrada para extrair a geometria
        geom = _parse_selection_polygon(i_selection_g_scatter_plot)
        
        if not geom:
            out_alert = "[ERRO] Faça uma seleção (lasso/box) no Overview antes de clicar em Build selection."
            out_alert_open = True
            return [
                selectedpoints,
                df_store_updated,
                s_unchecked_points,
                out_ctx,
                dash.no_update,
                out_input_save_csv,
                out_dropdown_opts,
                out_dropdown_val,
                out_vars_style,
                json.dumps(allb_store),
                _format_allb_page_info(allb_store),
                out_alert,
                out_alert_open,
                out_msg,
            ]
                    # classes visíveis direto do figure (mais robusto)
        visible_classes = []
        if s_g_scatter_plot_figure and "data" in s_g_scatter_plot_figure:
            for tr in s_g_scatter_plot_figure["data"]:
                if tr.get("showlegend", False):
                    nm = tr.get("name", "")
                    vis = tr.get("visible", True)
                    if nm and vis not in ("legendonly", False):
                        visible_classes.append(nm)


        page_size = int(max(100, min(_safe_int(s_allb_page_size, 2000), MAX_ALLB_PAGE_SIZE)))
        
        spec = dict(
            geom=geom,
            class_filter=visible_classes if visible_classes else None,
            probs_rng=s_probs_rng if isinstance(s_probs_rng, (list, tuple)) and len(s_probs_rng) == 2 else None,
            vars_rng=s_vars_rng if isinstance(s_vars_rng, (list, tuple)) and len(s_vars_rng) == 2 else None,
            hide_relab=(s_hide_relab_val == [" Hide relabeled images"]),
        )

        ver = int(out_ctx.get("overview_version", 0))
        token, total = _allb_build_selection(
            project_name=str(out_ctx.get("project_name", "")), 
            csv_folder=str(out_ctx.get("csv_folder", "")), 
            csv_candidates=csv_candidates, 
            spec=spec, 
            version=ver,
            visible_classes=visible_classes  # ← ESSENCIAL: passar classes visíveis
        )

        allb_store = {"active": True, "token": token, "page": 0, "page_size": page_size, "total": int(total)}
        out_ctx["allb_active"] = True

        if total <= 0:
            df_store_updated = pd.DataFrame().to_json()
            out_alert = f"[OK] All-batches selection construída, mas não há pontos para essa seleção/filtros. Classes visíveis: {visible_classes}"
            out_alert_open = True
        else:
            df_page = _allb_make_page_df(token, 0, page_size, out_ctx)
            df_store_updated = df_page.to_json()
            out_alert = f"[OK] All-batches selection construída. total={total} | Classes filtradas: {len(visible_classes)} | {_format_allb_page_info(allb_store)}"
            out_alert_open = True

        selectedpoints = json.dumps([])
        s_unchecked_points = json.dumps([])
        out_header = "Image Labeling Tool - OVERVIEW (allb active)"

    # =========================
    # Paginação all-batches
    # =========================
    elif flag_callback in ("button_allb_prev", "button_allb_next"):
        if not allb_store.get("active") or not allb_store.get("token"):
            out_alert = "[ERRO] All-batches selection não está ativo. Use Build selection primeiro."
            out_alert_open = True
        else:
            token = str(allb_store.get("token"))
            total = _allb_total(token)
            page_size = int(allb_store.get("page_size", 2000))
            page = int(allb_store.get("page", 0))

            if flag_callback == "button_allb_prev":
                page = max(0, page - 1)
            else:
                max_page = max(0, (total - 1) // page_size) if total > 0 else 0
                page = min(max_page, page + 1)

            allb_store["page"] = int(page)
            allb_store["total"] = int(total)

            out_ctx["is_overview"] = True
            out_ctx["allb_active"] = True

            df_page = _allb_make_page_df(token, page, page_size, out_ctx) if total > 0 else pd.DataFrame()
            df_store_updated = df_page.to_json()

            selectedpoints = json.dumps([])
            s_unchecked_points = json.dumps([])

            out_alert = f"[OK] {_format_allb_page_info(allb_store)}"
            out_alert_open = True
            out_header = "Image Labeling Tool - OVERVIEW (allb active)"

    # =========================
    # Aplicar label manual (botão Label)
    # =========================
    elif flag_callback == "button_aplicar_novo_label":
        new_label = str(s_input_aplicar_novo_label_value) if s_input_aplicar_novo_label_value is not None else ""
        if not new_label.strip():
            out_msg = "[ERRO] Label vazio."
        else:
            selected_and_checked = []
            selected_not_checked = []
            if s_button_galeria_filtrar_images:
                for point in s_button_galeria_filtrar_images:
                    cd = point.get("custom_data")
                    if cd is None:
                        continue
                    if point.get("isSelected") is True:
                        selected_and_checked.append(str(cd))
                    else:
                        selected_not_checked.append(str(cd))

            # === ALLB MODE: custom_data é global_id => atualiza múltiplos CSVs ===
            if bool(out_ctx.get("allb_active", False)):
                token = _current_allb_token()
                changed_batches = set()

                # undo snapshot
                undo_batches = {}
                undo_page = df_updated.copy()

                # agrupa por batch
                by_batch: dict[str, list[str]] = {}
                for gid in selected_and_checked:
                    sp = _split_global_id(gid)
                    if not sp:
                        continue
                    bn, cd_orig = sp
                    by_batch.setdefault(bn, []).append(cd_orig)

                for bn, cds in by_batch.items():
                    csv_path = join(str(out_ctx.get("csv_folder", "")), bn)
                    if not exists(csv_path):
                        continue

                    # df do batch (cache se existir; senão lê full e coloca no cache APENAS se for modificar)
                    if bn in BATCH_DF_CACHE:
                        dfb = BATCH_DF_CACHE[bn].copy()
                    else:
                        dfb = _ensure_required_columns(read_input_csv(csv_path))

                    # guarda undo
                    undo_batches[bn] = dfb.copy()

                    # aplica
                    col_id = int(_ensure_color_for_label(new_label))
                    m = dfb["custom_data"].astype(str).isin([str(x) for x in cds])
                    if m.any():
                        dfb.loc[m, "manual_label"] = new_label
                        dfb.loc[m, "colors"] = col_id
                        BATCH_DF_CACHE[bn] = dfb.copy()
                        DIRTY_BATCHES.add(bn)
                        changed_batches.add(bn)

                # atualiza page df
                col_id = int(_ensure_color_for_label(new_label))
                df_updated.loc[df_updated["custom_data"].isin(selected_and_checked), "manual_label"] = new_label
                if "colors" in df_updated.columns:
                    df_updated.loc[df_updated["custom_data"].isin(selected_and_checked), "colors"] = col_id

                # registra undo
                UNDO_SNAPSHOT = {"mode": "allb", "batches": undo_batches, "page_df": undo_page}

                # salva (opcional)
                if s_check_save_csv_value == [" Save CSV after labeling"]:
                    for bn in sorted(changed_batches):
                        try:
                            save_csv(BATCH_DF_CACHE[bn], bn)
                        except Exception:
                            pass

                # bump overview + refresh selection (recalcula só batches alterados)
                _bump_overview_version()
                if token and changed_batches:
                    _refresh_allb_after_batch_changes(sorted(changed_batches))

                df_store_updated = df_updated.to_json()
                selectedpoints = json.dumps([])

            # === NORMAL BATCH MODE ===
            else:
                # undo snapshot (batch)
                UNDO_SNAPSHOT = {"mode": "batch", "csv_basename": str(out_ctx.get("csv_basename", "")), "df_before": df_updated.copy()}

                col_id = int(_ensure_color_for_label(new_label))
                df_updated.loc[df_updated["custom_data"].isin(selected_and_checked), "manual_label"] = new_label
                df_updated.loc[df_updated["custom_data"].isin(selected_and_checked), "colors"] = col_id

                csv_bn = str(out_ctx.get("csv_basename", ""))
                if csv_bn and csv_bn != OVERVIEW_VALUE:
                    BATCH_DF_CACHE[csv_bn] = df_updated.copy()
                    DIRTY_BATCHES.add(csv_bn)

                _bump_overview_version()

                if s_check_save_csv_value == [" Save CSV after labeling"]:
                    save_csv(df_updated, str(s_input_save_csv_value))

                df_store_updated = df_updated.to_json()

                if s_check_discard_marked_value == [" Discard images after labeling"]:
                    selectedpoints = json.dumps(selected_not_checked)

    # =========================
    # Relabel by cluster (apenas imagens atualmente selecionadas/marcadas)
    # =========================
    elif flag_callback == "button_relabel_cluster":
        selected_and_checked = []
        selected_not_checked = []
        if s_button_galeria_filtrar_images:
            for point in s_button_galeria_filtrar_images:
                cd = point.get("custom_data")
                if cd is None:
                    continue
                if point.get("isSelected") is True:
                    selected_and_checked.append(str(cd))
                else:
                    selected_not_checked.append(str(cd))

        try:
            meta = json.loads(s_cluster_meta_json) if s_cluster_meta_json else {}
        except Exception:
            meta = {}

        if not meta or not meta.get("ready"):
            out_msg = "[ERRO] Cluster meta não está pronto. Clique primeiro em 'Update cluster (cut)'."
        else:
            class_label = str(meta.get("class", ""))
            mapping = meta.get("map", {}) if isinstance(meta.get("map", {}), dict) else {}

            if not class_label:
                out_msg = "[ERRO] Meta de cluster sem classe. Refaça 'Update cluster (cut)'."
            else:
                if bool(out_ctx.get("allb_active", False)):
                    try:
                        unchecked_list = json.loads(s_unchecked_points) if s_unchecked_points else []
                        unchecked_set = set(str(x) for x in unchecked_list if x is not None)
                    except Exception:
                        unchecked_set = set()

                    # aplica em TODOS os itens do cluster (todas as páginas)
                    selected_and_checked = [str(k) for k in mapping.keys() if str(k) not in unchecked_set]
                    selected_not_checked = []
                    
                # === ALLB MODE ===
                if bool(out_ctx.get("allb_active", False)):
                    token = _current_allb_token()
                    changed_batches = set()

                    # undo snapshot
                    undo_batches = {}
                    undo_page = df_updated.copy()

                    # agrupa por batch, aplicando label novo por cluster
                    by_batch: dict[str, list[tuple[str, str]]] = {}
                    # (cd_orig, new_lab)
                    for gid in selected_and_checked:
                        cid = mapping.get(gid, None)
                        if cid is None:
                            continue
                        sp = _split_global_id(gid)
                        if not sp:
                            continue
                        bn, cd_orig = sp
                        new_lab = f"{class_label}__g{int(cid)}"
                        by_batch.setdefault(bn, []).append((cd_orig, new_lab))

                    for bn, pairs in by_batch.items():
                        csv_path = join(str(out_ctx.get("csv_folder", "")), bn)
                        if not exists(csv_path):
                            continue

                        if bn in BATCH_DF_CACHE:
                            dfb = BATCH_DF_CACHE[bn].copy()
                        else:
                            dfb = _ensure_required_columns(read_input_csv(csv_path))

                        undo_batches[bn] = dfb.copy()

                        for cd_orig, new_lab in pairs:
                            col_id = int(_ensure_color_for_label(new_lab))
                            m = (dfb["custom_data"].astype(str) == str(cd_orig))
                            if m.any():
                                dfb.loc[m, "manual_label"] = new_lab
                                dfb.loc[m, "colors"] = col_id

                        BATCH_DF_CACHE[bn] = dfb.copy()
                        DIRTY_BATCHES.add(bn)
                        changed_batches.add(bn)

                    # atualiza page df
                    changed = 0
                    missing = 0
                    for gid in selected_and_checked:
                        cid = mapping.get(gid, None)
                        if cid is None:
                            missing += 1
                            continue
                        new_lab = f"{class_label}__g{int(cid)}"
                        df_updated.loc[df_updated["custom_data"] == gid, "manual_label"] = new_lab
                        if "colors" in df_updated.columns:
                            df_updated.loc[df_updated["custom_data"] == gid, "colors"] = int(_ensure_color_for_label(new_lab))
                        changed += 1

                    UNDO_SNAPSHOT = {"mode": "allb", "batches": undo_batches, "page_df": undo_page}

                    if s_check_save_csv_value == [" Save CSV after labeling"]:
                        for bn in sorted(changed_batches):
                            try:
                                save_csv(BATCH_DF_CACHE[bn], bn)
                            except Exception:
                                pass

                    _bump_overview_version()
                    if token and changed_batches:
                        _refresh_allb_after_batch_changes(sorted(changed_batches))

                    df_store_updated = df_updated.to_json()
                    if s_check_discard_marked_value == [" Discard images after labeling"]:
                        selectedpoints = json.dumps(selected_not_checked)

                    out_msg = f"[OK] Relabel by cluster (allb): alteradas={changed} | sem_cluster_id={missing}."

                # === NORMAL BATCH MODE ===
                else:
                    # undo snapshot
                    UNDO_SNAPSHOT = {"mode": "batch", "csv_basename": str(out_ctx.get("csv_basename", "")), "df_before": df_updated.copy()}

                    changed = 0
                    missing = 0
                    for cd in selected_and_checked:
                        cid = mapping.get(cd, None)
                        if cid is None:
                            missing += 1
                            continue
                        new_lab = f"{class_label}__g{int(cid)}"
                        df_updated.loc[df_updated["custom_data"] == cd, "manual_label"] = new_lab
                        df_updated.loc[df_updated["custom_data"] == cd, "colors"] = int(_ensure_color_for_label(new_lab))
                        changed += 1

                    csv_bn = str(out_ctx.get("csv_basename", ""))
                    if csv_bn and csv_bn != OVERVIEW_VALUE:
                        BATCH_DF_CACHE[csv_bn] = df_updated.copy()
                        DIRTY_BATCHES.add(csv_bn)

                    _bump_overview_version()

                    df_store_updated = df_updated.to_json()
                    if s_check_save_csv_value == [" Save CSV after labeling"]:
                        save_csv(df_updated, str(s_input_save_csv_value))
                    if s_check_discard_marked_value == [" Discard images after labeling"]:
                        selectedpoints = json.dumps(selected_not_checked)

                    out_msg = f"[OK] Relabel by cluster: alteradas={changed} | sem_cluster_id={missing}."

    # =========================
    # UNDO (última operação)
    # =========================
    elif flag_callback == "button_undo":
        if not UNDO_SNAPSHOT:
            out_msg = "[UNDO] Nada para desfazer."
        else:
            mode = UNDO_SNAPSHOT.get("mode")
            if mode == "batch":
                bn = str(UNDO_SNAPSHOT.get("csv_basename", ""))
                df_before = UNDO_SNAPSHOT.get("df_before", None)
                if isinstance(df_before, pd.DataFrame):
                    df_updated = df_before.copy()
                    df_store_updated = df_updated.to_json()
                    if bn and bn != OVERVIEW_VALUE:
                        BATCH_DF_CACHE[bn] = df_updated.copy()
                        DIRTY_BATCHES.add(bn)
                    _bump_overview_version()
                    out_msg = "[UNDO] Batch revertido."
                UNDO_SNAPSHOT = None
                selectedpoints = json.dumps([])

            elif mode == "allb":
                batches = UNDO_SNAPSHOT.get("batches", {})
                page_df = UNDO_SNAPSHOT.get("page_df", None)

                changed_batches = []
                if isinstance(batches, dict):
                    for bn, dfb in batches.items():
                        if isinstance(dfb, pd.DataFrame):
                            BATCH_DF_CACHE[str(bn)] = dfb.copy()
                            DIRTY_BATCHES.add(str(bn))
                            changed_batches.append(str(bn))

                if isinstance(page_df, pd.DataFrame):
                    df_updated = page_df.copy()
                    df_store_updated = df_updated.to_json()

                _bump_overview_version()
                token = _current_allb_token()
                if token and changed_batches:
                    _refresh_allb_after_batch_changes(sorted(set(changed_batches)))

                UNDO_SNAPSHOT = None
                selectedpoints = json.dumps([])
                out_msg = "[UNDO] All-batches revertido (última operação)."

    # =========================
    # Invert marks
    # =========================
    elif flag_callback == "button_invert_marks":
        if s_button_galeria_filtrar_images:
            for point in s_button_galeria_filtrar_images:
                if point.get("isSelected") is True:
                    cd = point.get("custom_data")
                    if cd is not None:
                        unchecked_points.append(str(cd))
        df_store_updated = df_updated.to_json()

    # --- outputs comuns ---
    s_unchecked_points_out = json.dumps([x for x in unchecked_points if x is not None]) if unchecked_points else (s_unchecked_points or json.dumps([]))
    allb_store_out = json.dumps(allb_store)
    allb_info = _format_allb_page_info(allb_store)

    return [
        selectedpoints,
        df_store_updated,
        s_unchecked_points_out,
        out_ctx,
        out_header,
        out_input_save_csv,
        out_dropdown_opts,
        out_dropdown_val,
        out_vars_style,
        allb_store_out,
        allb_info,
        out_alert,
        out_alert_open,
        out_msg,
    ]
    

# atualiza opções do dropdown de classe do cluster dinamicamente
@app.callback(
    Output("dropdown_cluster_class", "options"),
    Input("store_context", "data"),
    Input("state_store_df", "data"),
)
def refresh_class_options(s_ctx, s_store_df):
    s_ctx = s_ctx or {}

    # opções do cluster devem considerar o DF atual
    try:
        _df = pd.read_json(StringIO(s_store_df))
        labs = sorted(_df["manual_label"].astype(str).unique().tolist(), key=lambda s: s.lower()) if "manual_label" in _df.columns else []
    except Exception:
        labs = []

    opts_cluster = [{"label": x, "value": x} for x in labs]
    return opts_cluster


# cluster update (botão)  [FIX: allb = cluster em TODA a seleção, não só na página]
@app.callback(
    Output("cluster_meta_store", "data"),
    Output("graph_cluster_dendrogram", "figure"),
    Output("cluster_summary", "children"),
    Output("alert_cluster_problem", "children"),
    Output("alert_cluster_problem", "is_open"),
    Input("button_update_cluster", "n_clicks"),
    State("dropdown_cluster_class", "value"),
    State("slider_cluster_cut", "value"),
    State("state_store_df", "data"),
    State("selected_custom_points", "data"),
    State("probs_range_slider", "value"),
    State("vars_range_slider", "value"),
    State("check_discard_relabeled", "value"),
    State("store_context", "data"),
    # >>> ADICIONADOS:
    State("allb_store", "data"),
    State("unchecked_points", "data"),
)

def update_cluster_view(
    nclicks,
    class_label,
    cut01,
    s_store_df,
    s_selected_custom_points,
    probs_rng,
    vars_rng,
    hide_relab_val,
    s_ctx,
    s_allb_store_json,          # << novo
    s_unchecked_points_json,    # << novo
):
    if not nclicks:
        return json.dumps({"ready": False}), go.Figure(), "", "", False

    if linkage is None or fcluster is None:
        msg = "scipy (linkage/fcluster) não disponível. Instale scipy para clustering."
        fig = go.Figure()
        fig.add_annotation(text=msg, showarrow=False)
        return json.dumps({"ready": False}), fig, "", msg, True

    s_ctx = s_ctx or {}
    project = str(s_ctx.get("project_name", ""))
    allb_active = bool(s_ctx.get("allb_active", False))

    if not class_label:
        return json.dumps({"ready": False}), go.Figure(), "", "Selecione uma classe primeiro.", True

    # --- parse allb_store/token ---
    try:
        allb_store = json.loads(s_allb_store_json) if s_allb_store_json else {}
    except Exception:
        allb_store = {}

    token = None
    if allb_active and isinstance(allb_store, dict) and allb_store.get("active") and allb_store.get("token"):
        token = str(allb_store.get("token"))

    # --- unchecked_set (para respeitar invert/uncheck across pages) ---
    try:
        unchecked_points = json.loads(s_unchecked_points_json) if s_unchecked_points_json else []
        unchecked_points = [str(x) for x in unchecked_points if x is not None]
    except Exception:
        unchecked_points = []
    unchecked_set = set(unchecked_points)

    # ------------------------------------------------------------
    # BASE SELECTION:
    # - allb_active + token: usa TODA a seleção do token (todas as páginas)
    # - caso contrário: usa seleção do scatter (modo batch normal)
    # ------------------------------------------------------------
    if allb_active and token:
        cache = ALLB_CACHE.get(token)
        if not cache:
            return json.dumps({"ready": False}), go.Figure(), "", "All-batches token inválido/expirado. Refaça Build selection.", True

        csv_folder_local = str(s_ctx.get("csv_folder", ""))
        batch_order = cache.get("batch_order", [])
        packmasks = cache.get("packmasks", {})
        lens = cache.get("lens", {})

        rows = []
        for bn in batch_order:
            pack = packmasks.get(bn, None)
            n = int(lens.get(bn, 0))
            if pack is None or n <= 0:
                continue

            mask_bits = np.unpackbits(np.frombuffer(pack, dtype=np.uint8), count=n).astype(bool)
            idx_sel = np.flatnonzero(mask_bits)
            if idx_sel.size == 0:
                continue

            csv_path = join(csv_folder_local, bn)
            if not exists(csv_path):
                continue

            try:
                dfb = _read_batch_minimal(csv_path, bn)
            except Exception:
                continue

            if dfb is None or dfb.empty:
                continue

            idx_sel = idx_sel[idx_sel < len(dfb)]
            if idx_sel.size == 0:
                continue

            part = dfb.iloc[idx_sel].copy()

            # filtra a classe alvo (antes dos filtros numéricos)
            if "manual_label" not in part.columns:
                continue
            part = part[part["manual_label"].astype(str) == str(class_label)].copy()
            if part.empty:
                continue

            # converte custom_data -> global_id (compatível com Apply Cluster Labels no allb)
            part["csv_basename"] = bn
            part["custom_data_orig"] = part["custom_data"].astype(str)
            part["custom_data"] = [_global_id(bn, cd) for cd in part["custom_data_orig"].astype(str).tolist()]

            # remove os desmarcados (invert/uncheck)
            if unchecked_set:
                part = part[~part["custom_data"].astype(str).isin(unchecked_set)].copy()
                if part.empty:
                    continue

            rows.append(part)

        if not rows:
            return json.dumps({"ready": False}), go.Figure(), "", "Nenhum ponto disponível nessa classe (allb selection total).", True

        df_sel = pd.concat(rows, ignore_index=True)

    else:
        # --- modo batch normal (igual ao seu, baseado na seleção do scatter) ---
        try:
            _df = pd.read_json(StringIO(s_store_df))
        except Exception as e:
            return json.dumps({"ready": False}), go.Figure(), "", f"Falha ao ler DF: {e}", True

        for c in ["custom_data", "manual_label", "correct_label"]:
            if c in _df.columns:
                _df[c] = _df[c].astype(str)

        if "x" not in _df.columns or "y" not in _df.columns:
            return json.dumps({"ready": False}), go.Figure(), "", "DF não tem colunas x/y.", True

        selected_points = json.loads(s_selected_custom_points) if s_selected_custom_points else []
        selected_points = [str(x) for x in selected_points if x is not None]
        if not selected_points:
            return json.dumps({"ready": False}), go.Figure(), "", "Selecione pontos no scatter primeiro.", True

        df_sel = _df.loc[_df["custom_data"].isin(selected_points)].copy()
        df_sel = df_sel[df_sel["manual_label"].astype(str) == str(class_label)].copy()

        if df_sel.empty:
            return json.dumps({"ready": False}), go.Figure(), "", "Nenhum ponto disponível nessa classe (subset atual).", True

    # ------------------------------------------------------------
    # FILTROS (mantém sua lógica)
    # ------------------------------------------------------------
    if "confs1" in df_sel.columns and probs_rng is not None:
        df_sel["confs1"] = pd.to_numeric(df_sel["confs1"], errors="coerce")
        df_sel = df_sel[(df_sel["confs1"] >= probs_rng[0]) & (df_sel["confs1"] <= probs_rng[1])]

    has_var = bool(s_ctx.get("has_var", False))
    if has_var and "var_norm" in df_sel.columns and vars_rng is not None:
        df_sel["var_norm"] = pd.to_numeric(df_sel["var_norm"], errors="coerce")
        df_sel = df_sel[(df_sel["var_norm"] >= vars_rng[0]) & (df_sel["var_norm"] <= vars_rng[1])]

    if hide_relab_val == [" Hide relabeled images"]:
        if "correct_label" in df_sel.columns:
            df_sel = df_sel[df_sel["manual_label"].astype(str) == df_sel["correct_label"].astype(str)]

    # saneamento final
    df_sel["x"] = pd.to_numeric(df_sel["x"], errors="coerce")
    df_sel["y"] = pd.to_numeric(df_sel["y"], errors="coerce")
    df_sel = df_sel.dropna(subset=["x", "y", "custom_data"])

    N = int(df_sel.shape[0])
    if N < 2:
        return json.dumps({"ready": False}), go.Figure(), "", "Poucos pontos após filtros (N<2).", True
    if N > MAX_CLUSTER_N:
        return json.dumps({"ready": False}), go.Figure(), "", f"N={N} > MAX_CLUSTER_N={MAX_CLUSTER_N}. Aumente MAX_CLUSTER_N ou reduza filtros.", True

    X = df_sel[["x", "y"]].to_numpy(dtype=np.float64)

    try:
        Z = linkage(X, method="average", metric="euclidean")
    except Exception as e:
        return json.dumps({"ready": False}), go.Figure(), "", f"Falha linkage: {type(e).__name__}: {e}", True

    dist_max = float(np.max(Z[:, 2])) if Z.size else 1.0
    if not np.isfinite(dist_max) or dist_max <= 0:
        dist_max = 1.0

    cut01 = float(cut01) if cut01 is not None else 0.50
    cut01 = min(max(cut01, 0.0), 1.0)
    cut_dist = cut01 * dist_max

    try:
        labels = fcluster(Z, t=cut_dist, criterion="distance")
    except Exception as e:
        return json.dumps({"ready": False}), go.Figure(), "", f"Falha fcluster: {type(e).__name__}: {e}", True

    K = int(np.max(labels)) if labels.size else 0

    # mapping precisa bater com o "Apply Cluster Labels"
    # - batch normal: custom_data normal
    # - allb: custom_data = global_id (csv::custom_data_orig)
    cds = df_sel["custom_data"].astype(str).tolist()
    mapping = {cd: int(cid) for cd, cid in zip(cds, labels.tolist())}

    try:
        fig = _build_dendrogram_figure(Z, cut_dist=cut_dist)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Dendrograma falhou: {type(e).__name__}: {e}", showarrow=False)

    scope = "allb(selection TOTAL)" if (allb_active and token) else "seleção no scatter"
    summary = (
        f"[cluster] class='{class_label}' | N={N} | cut={cut01:.2f} "
        f"(dist={cut_dist:.6g}/{dist_max:.6g}) | groups={K}\n"
        f"Obs: clustering é calculado em: {scope} + filtros."
    )

    meta = {
        "ready": True,
        "project": project,
        "class": str(class_label),
        "cut01": float(cut01),
        "cut_dist": float(cut_dist),
        "dist_max": float(dist_max),
        "N": int(N),
        "groups": int(K),
        "map": mapping,
        "scope": "allb" if (allb_active and token) else "batch",
        "token": token,
    }
    return json.dumps(meta), fig, summary, "", False



@app.callback(
    [
        Output("g_scatter_plot", "figure"),
        Output("g_image_selector", "images"),
    ],
    [
        Input("state_store_df", "data"),
        Input("selected_custom_points", "data"),
        Input("unchecked_points", "data"),
        Input("dropdown_order_labels", "value"),
        Input("slider_marker_size", "value"),
        Input("probs_range_slider", "value"),
        Input("vars_range_slider", "value"),
        Input("dropdown_order_images", "value"),
        Input("check_marked_first", "value"),
        Input("check_discard_relabeled", "value"),
        Input("store_context", "data"),
        Input("allb_store", "data"),
    ],
    [
        State("g_scatter_plot", "figure"),
        State("g_image_selector", "images"),
    ],
)


def scatter_plot_image_selector(
    i_store_df,
    i_selected_custom_points,
    i_unchecked_points,
    i_dropdown_order_labels_value,
    i_slider_marker_size_value,
    i_range_slider_probs,
    i_range_slider_vars,
    i_dropdown_order_images_value,
    i_check_marked_first_value,
    i_check_discard_relabeled_value,
    s_ctx,
    s_allb_store_json,
    s_g_scatter_plot_figure,
    s_g_image_selector_images,
):
    s_ctx = s_ctx or {}
    is_overview = bool(s_ctx.get("is_overview", False))
    allb_active = bool(s_ctx.get("allb_active", False))

    # tenta ler df atual
    try:
        _df = pd.read_json(StringIO(i_store_df))
    except Exception:
        _df = pd.DataFrame()

    for c in ["custom_data", "names", "manual_label", "correct_label", "csv_basename", "custom_data_orig"]:
        if c in _df.columns:
            _df[c] = _df[c].astype(str)

    # unchecked
    unchecked_points = json.loads(i_unchecked_points) if i_unchecked_points else []
    unchecked_points = [str(x) for x in unchecked_points if x is not None]
    unchecked_set = set(unchecked_points)

    # ====== OVERVIEW MODE ======
    if is_overview:
        ver = int(s_ctx.get("overview_version", 0))
        proj = str(s_ctx.get("project_name", project_name))
        folder = str(s_ctx.get("csv_folder", csv_folder))
        fig_over = _get_overview_fig_cached(proj, folder, version=ver, graph_range=graph_range)

        # se allb não está ativo => sem galeria
        if not allb_active:
            fig2 = create_list_dics(_list_src=[], _list_thumbnail=[], _list_name_figure=[])
            return [fig_over, fig2]

        # allb ativo => galeria baseada no page DF (_df)
        if _df is None or _df.empty:
            fig2 = create_list_dics(_list_src=[], _list_thumbnail=[], _list_name_figure=[])
            return [fig_over, fig2]

        # ordem multi-critério (somente dentro da página)
        order_vals = i_dropdown_order_images_value
        if isinstance(order_vals, str):
            order_vals = [order_vals]
        if not order_vals:
            order_vals = ["A-Z, a-z"]

        filtered_df = _df.copy()
        # (nota: filtros de conf/var/hide_relab já foram aplicados no build selection; aqui só respeitamos hide_relab por segurança)
        if i_check_discard_relabeled_value == [" Hide relabeled images"] and "correct_label" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["manual_label"].astype(str) == filtered_df["correct_label"].astype(str)]

        sort_keys, ascending = [], []
        for crit in order_vals:
            if crit == "Similarity" and "D7" in filtered_df.columns:
                sort_keys.append("D7"); ascending.append(True); continue
            if crit == "Confidence" and "confs1" in filtered_df.columns:
                sort_keys.append("confs1"); ascending.append(False); continue
            if crit == "Variance" and "var_norm" in filtered_df.columns:
                sort_keys.append("var_norm"); ascending.append(False); continue
            if crit == "Conf diff" and ("confs1" in filtered_df.columns) and ("confs2" in filtered_df.columns):
                if "conf_diff" not in filtered_df.columns:
                    filtered_df = filtered_df.copy()
                    filtered_df["conf_diff"] = pd.to_numeric(filtered_df["confs1"], errors="coerce") - pd.to_numeric(filtered_df["confs2"], errors="coerce")
                sort_keys.append("conf_diff"); ascending.append(False); continue
            if crit == "A-Z, a-z" and "manual_label" in filtered_df.columns:
                sort_keys.append("manual_label"); ascending.append(True); continue
            if crit in map_columns_vals.values():
                key = [k for k, v in map_columns_vals.items() if v == crit][0]
                if key in filtered_df.columns:
                    sort_keys.append(key); ascending.append(False)
                continue

        if not sort_keys:
            sort_keys = ["manual_label"] if "manual_label" in filtered_df.columns else ["names"]
            ascending = [True]

        filtered_df = filtered_df.sort_values(by=sort_keys, ascending=ascending, kind="mergesort").copy()

        # preserva marcações (sem repetição entre páginas: cada página tem seus itens únicos)
        gallery_map = {}
        if s_g_image_selector_images:
            for it in s_g_image_selector_images:
                cd = it.get("custom_data")
                if cd is not None:
                    gallery_map[str(cd)] = bool(it.get("isSelected", True))

        # default: True, exceto se já foi invertido (unchecked_set) ou estado anterior
        sel_flags = []
        for cd in filtered_df["custom_data"].astype(str).tolist():
            if cd in unchecked_set:
                sel_flags.append(False)
            else:
                sel_flags.append(gallery_map.get(cd, True))

        filtered_df["selected"] = sel_flags

        if i_check_marked_first_value == [" Marked images first"]:
            checked_df = filtered_df[filtered_df["selected"] == True]
            unchecked_df = filtered_df[filtered_df["selected"] == False]
            filtered_df = pd.concat([checked_df, unchecked_df], ignore_index=True)

        # paths por linha (por batch)
        srcs = []
        thumbs = []
        for _, row in filtered_df.iterrows():
            bn = str(row.get("csv_basename", ""))
            name = str(row.get("names", ""))
            img_dir = _images_dir_for_csv(s_ctx, bn) if bn else str(s_ctx.get("path_to_images", ""))
            srcs.append(_abs_to_assets_url(join(img_dir, name)))
            thumbs.append(_abs_to_assets_url(join(img_dir, name)))

        # captions
        _image_names = filtered_df["names"].astype(str).tolist()
        _image_correct = filtered_df["correct_label"].astype(str).tolist() if "correct_label" in filtered_df.columns else [""] * len(filtered_df)
        _image_attributes = [""] * len(filtered_df)

        first_crit = order_vals[0] if len(order_vals) > 0 else ""
        if first_crit in map_columns_vals.values():
            key = [k for k, v in map_columns_vals.items() if v == first_crit][0]
            if key in filtered_df.columns:
                _image_attributes = filtered_df[key].tolist()

        if "preds1" not in filtered_df.columns:
            texts = [f"{n[:-4]} - Pred: {lab}, Attr: {val}" for n, lab, val in zip(_image_names, _image_correct, _image_attributes)]
        elif "preds2" not in filtered_df.columns:
            p1 = filtered_df["preds1"]; c1 = filtered_df["confs1"] if "confs1" in filtered_df.columns else [0]*len(filtered_df)
            texts = [f"{n[:-4]} - {pp1} ({100*float(cc1):.1f}%, Attr: {val})" for pp1, cc1, n, val in zip(p1, c1, _image_names, _image_attributes)]
        else:
            p1 = filtered_df["preds1"]; c1 = filtered_df["confs1"]
            p2 = filtered_df["preds2"]; c2 = filtered_df["confs2"]
            p3 = filtered_df["preds3"]; c3 = filtered_df["confs3"]
            texts = [
                f"{n[:-4]} - {pp1} ({100*float(cc1):.1f}%, Attr: {val}), {pp2} ({100*float(cc2):.1f}%), {pp3} ({100*float(cc3):.1f}%)"
                for pp1, cc1, pp2, cc2, pp3, cc3, n, val in zip(p1, c1, p2, c2, p3, c3, _image_names, _image_attributes)
            ]

        widths = filtered_df["widths"].tolist() if "widths" in filtered_df.columns else [1]*len(filtered_df)
        heights = filtered_df["heights"].tolist() if "heights" in filtered_df.columns else [1]*len(filtered_df)

        fig2 = create_list_dics(
            _list_src=srcs,
            _list_thumbnail=thumbs,
            _list_name_figure=_image_names,
            _list_thumbnailWidth=widths,
            _list_thumbnailHeight=heights,
            _list_isSelected=list(filtered_df["selected"].astype(bool).tolist()),
            _list_custom_data=filtered_df["custom_data"].astype(str).tolist(),
            _list_caption=texts,
            _list_thumbnailCaption=filtered_df["manual_label"].astype(str).tolist() if "manual_label" in filtered_df.columns else [""]*len(filtered_df),
            _list_tags=[[]] * filtered_df.shape[0],
        )

        return [fig_over, fig2]

    # ====== NORMAL BATCH MODE ======
    prev_fig = go.Figure(s_g_scatter_plot_figure)

    path_to_images_fs = s_ctx.get("path_to_images", "")
    path_to_thumbnails_fs = s_ctx.get("path_to_thumbnails", path_to_images_fs)
    background_path = s_ctx.get("background_path", BLANK_BG_PATH)
    thumbs_jpg_local = bool(s_ctx.get("thumbs_jpg", False))
    has_var_local = bool(s_ctx.get("has_var", False))

    selected_points = json.loads(i_selected_custom_points) if i_selected_custom_points else []
    selected_points = [str(x) for x in selected_points if x is not None]

    # ordem de labels
    label_order = None
    if "manual_label" in _df.columns:
        labs = _df["manual_label"].astype(str)
        if i_dropdown_order_labels_value == "Frequency":
            label_order = labs.value_counts().index.tolist()
        else:
            label_order = sorted(labs.unique().tolist(), key=lambda s: s.lower())

    fig_scatter = f_figure_scatter_plot(
        _df,
        _columns=["x", "y"],
        _selected_custom_data=selected_points,
        prev_fig=prev_fig,
        order_by=i_dropdown_order_labels_value,
        background_img=_bg_or_blank(background_path),
        opacity_changed=False,
        opacity=FIXED_BACKGROUND_OPACITY,
        marker_size=i_slider_marker_size_value,
        xrange=background_ranges["x"],
        yrange=background_ranges["y"],
    )

    if label_order is not None:
        try:
            data_sorted = sorted(
                list(fig_scatter.data),
                key=lambda tr: label_order.index(str(tr.name)) if str(tr.name) in label_order else 10**9,
            )
            fig_scatter.data = tuple(data_sorted)
        except Exception:
            pass

    filtered_df = _df.loc[_df["custom_data"].isin(selected_points)] if "custom_data" in _df.columns else pd.DataFrame()

    if "confs1" in filtered_df.columns and i_range_slider_probs is not None:
        filtered_df["confs1"] = pd.to_numeric(filtered_df["confs1"], errors="coerce")
        filtered_df = filtered_df.loc[
            (filtered_df["confs1"] >= i_range_slider_probs[0]) & (filtered_df["confs1"] <= i_range_slider_probs[1])
        ]

    if has_var_local and "var_norm" in filtered_df.columns and i_range_slider_vars is not None:
        filtered_df["var_norm"] = pd.to_numeric(filtered_df["var_norm"], errors="coerce")
        filtered_df = filtered_df.loc[
            (filtered_df["var_norm"] >= i_range_slider_vars[0]) & (filtered_df["var_norm"] <= i_range_slider_vars[1])
        ]

    if i_check_discard_relabeled_value == [" Hide relabeled images"]:
        if "correct_label" in filtered_df.columns and "manual_label" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["manual_label"] == filtered_df["correct_label"]]

    if len(selected_points) == 0 or filtered_df.shape[0] == 0:
        fig2 = create_list_dics(_list_src=[], _list_thumbnail=[], _list_name_figure=[])
        return [fig_scatter, fig2]

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
                sort_keys.append(key); ascending.append(False)
            continue

        if crit == "Similarity" and "D7" in filtered_df.columns:
            sort_keys.append("D7"); ascending.append(True); continue

        if crit == "Confidence" and "confs1" in filtered_df.columns:
            sort_keys.append("confs1"); ascending.append(False); continue

        if crit == "Variance" and "var_norm" in filtered_df.columns:
            sort_keys.append("var_norm"); ascending.append(False); continue

        if crit == "Conf diff" and ("confs1" in filtered_df.columns) and ("confs2" in filtered_df.columns):
            if "conf_diff" not in filtered_df.columns:
                filtered_df = filtered_df.copy()
                filtered_df["conf_diff"] = filtered_df["confs1"] - filtered_df["confs2"]
            sort_keys.append("conf_diff"); ascending.append(False); continue

        if crit == "A-Z, a-z" and "manual_label" in filtered_df.columns:
            sort_keys.append("manual_label"); ascending.append(True); continue

        if crit in filtered_df.columns:
            sort_keys.append(crit); ascending.append(True)

    if not sort_keys:
        sort_keys = ["manual_label"] if "manual_label" in filtered_df.columns else ["names"]
        ascending = [True]

    ordered_df = filtered_df.sort_values(by=sort_keys, ascending=ascending, kind="mergesort").copy()
    ordered_df["custom_data"] = ordered_df["custom_data"].astype(str)

    gallery_map = {}
    if s_g_image_selector_images:
        for it in s_g_image_selector_images:
            cd = it.get("custom_data")
            if cd is not None:
                gallery_map[str(cd)] = bool(it.get("isSelected", True))

    selection_temp = []
    for cd in selected_points:
        if cd in unchecked_set:
            selection_temp.append(False)
        else:
            selection_temp.append(gallery_map.get(cd, True))

    df_selection = pd.DataFrame({"custom_data": [str(x) for x in selected_points], "selected": selection_temp})
    ordered_df = ordered_df.merge(df_selection, how="left", on="custom_data")
    ordered_df["selected"] = ordered_df["selected"].fillna(True).astype(bool)

    if i_check_marked_first_value == [" Marked images first"]:
        checked_df = ordered_df[ordered_df["selected"] == True]
        unchecked_df = ordered_df[ordered_df["selected"] == False]
        ordered_df = pd.concat([checked_df, unchecked_df])

    _image_names = ordered_df["names"].astype(str)
    _image_correct = ordered_df["correct_label"].astype(str) if "correct_label" in ordered_df.columns else [""] * ordered_df.shape[0]

    if thumbs_jpg_local and ("thumbnails" in ordered_df.columns):
        _image_thumbs = ordered_df["thumbnails"].astype(str)
    else:
        _image_thumbs = ordered_df["names"].astype(str)

    _image_widths = ordered_df["widths"] if "widths" in ordered_df.columns else [1] * ordered_df.shape[0]
    _image_heights = ordered_df["heights"] if "heights" in ordered_df.columns else [1] * ordered_df.shape[0]

    _image_attributes = [""] * ordered_df.shape[0]
    first_crit = order_vals[0] if len(order_vals) > 0 else ""
    if first_crit in map_columns_vals.values():
        key = [k for k, v in map_columns_vals.items() if v == first_crit][0]
        if key in ordered_df.columns:
            _image_attributes = ordered_df[key].tolist()

    if "preds1" in ordered_df.columns:
        p1 = ordered_df["preds1"]; c1 = ordered_df["confs1"] if "confs1" in ordered_df.columns else [0]*ordered_df.shape[0]
    if "preds2" in ordered_df.columns:
        p2 = ordered_df["preds2"]; c2 = ordered_df["confs2"]
        p3 = ordered_df["preds3"]; c3 = ordered_df["confs3"]

    if "preds1" not in ordered_df.columns:
        texts = [f"{name[:-4]} - Pred: {lab}, Attr: {val}" for name, lab, val in zip(_image_names, _image_correct, _image_attributes)]
    elif "preds2" not in ordered_df.columns:
        texts = [f"{name[:-4]} - {pp1} ({100*float(cc1):.1f}%, Attr: {val})" for pp1, cc1, name, val in zip(p1, c1, _image_names, _image_attributes)]
    else:
        texts = [
            f"{name[:-4]} - {pp1} ({100*float(cc1):.1f}%, Attr: {val}), {pp2} ({100*float(cc2):.1f}%), {pp3} ({100*float(cc3):.1f}%)"
            for pp1, cc1, pp2, cc2, pp3, cc3, name, val in zip(p1, c1, p2, c2, p3, c3, _image_names, _image_attributes)
        ]

    fig2 = create_list_dics(
        _list_src=[_abs_to_assets_url(join(path_to_images_fs, n)) for n in _image_names.tolist()],
        _list_thumbnail=[_abs_to_assets_url(join(path_to_thumbnails_fs, t)) for t in _image_thumbs.tolist()],
        _list_name_figure=_image_names.tolist(),
        _list_thumbnailWidth=list(_image_widths) if not hasattr(_image_widths, "tolist") else _image_widths.tolist(),
        _list_thumbnailHeight=list(_image_heights) if not hasattr(_image_heights, "tolist") else _image_heights.tolist(),
        _list_isSelected=list(ordered_df["selected"]),
        _list_custom_data=ordered_df["custom_data"].astype(str).tolist(),
        _list_caption=texts,
        _list_thumbnailCaption=ordered_df["manual_label"].astype(str).tolist() if "manual_label" in ordered_df.columns else [""]*ordered_df.shape[0],
        _list_tags=[[]] * ordered_df.shape[0],
    )

    return [fig_scatter, fig2]


# =========================
# RUN
# =========================
if __name__ == "__main__":
    _opened = False
    if not _opened:
        webbrowser.open("http://127.0.0.1:" + str(port) + "/", new=2, autoraise=True)
        _opened = True
    
    app.title = "IAT - Image Annotation Tool"
    print(f"\nServidor iniciado em: http://127.0.0.1:{port}")
    print("Pressione Ctrl+C para parar o servidor.\n")
    app.run_server(debug=False, port=port)
