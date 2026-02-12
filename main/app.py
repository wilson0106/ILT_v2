# app.py  (PARTE 1/5)
import sys
import re
import json
import shutil
import webbrowser
import tempfile
import uuid
import subprocess
from bisect import bisect_right
from datetime import date
from os import mkdir, listdir
from os.path import exists, join, dirname, basename, isdir, splitext
from io import StringIO
from urllib.parse import quote
from pathlib import Path
from threading import RLock

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

# --- clustering (subset-based / artifacts) ---
try:
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
except Exception:
    linkage = None
    fcluster = None
    dendrogram = None
# --- sklearn (PCA opcional p/ replicar pipeline do class_clustering.py) ---
try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None


try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None


# =========================
# CONFIG
# =========================
FIXED_BACKGROUND_OPACITY = 0.35
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

ASSETS_FOLDER = join(dirname(__file__), "assets")
ASSETS_URL_PATH = "/assets"

BLANK_BG_PATH = join(tempfile.gettempdir(), "iat_blank_bg.png")
OVERVIEW_VALUE = "__OVERVIEW__"

# limites (segurança/performance)
MAX_CLUSTER_N = 2500  # (ainda usado em alguns caminhos legacy; clustering por .npz não usa este cap)
MAX_OVERVIEW_POINTS_PER_CLASS = 20000
MAX_OVERVIEW_SELECTOR_TOTAL = 120000
MAX_ALLB_PAGE_SIZE = 8000

# dendrograma (novo)
DENDRO_TRUNCATE_AT = 2500     # acima disso desenha truncado
DENDRO_TRUNCATE_P = 200       # quantos "blocos/folhas" no truncado
DENDRO_HARD_CAP = 200000      # acima disso: não desenha (evita travar)

# caches server-side
BATCH_DF_CACHE: dict[str, pd.DataFrame] = {}
CSV_COLS_CACHE: dict[str, list[str]] = {}
BATCH_BBOX_CACHE: dict[str, tuple[float, float, float, float]] = {}

OVERVIEW_CACHE = {"project": None, "version": None, "fig": None}

ALLB_CACHE: dict[str, dict] = {}
DIRTY_BATCHES: set[str] = set()

UNDO_SNAPSHOT = None  # dict

# lock p/ callbacks (evita race)
_CACHE_LOCK = RLock()

# cluster artifacts cache
CLUSTERS_SUBDIR = "clusters"
CLUSTER_ARTIFACT_CACHE = {}  # (project, class_sanitized) -> {"mtime": float, "data": dict}


# =========================
# SMALL UTILS
# =========================
def _bg_or_blank(bg_path: str | None) -> str:
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
    if not p:
        return p

    if not isdir(p):
        base = basename(p).lower()
        parent = dirname(p)
        if base in ("samples", "sample") and isdir(parent):
            return parent
        return p

    if _dir_has_images(p):
        return p

    base = basename(p).lower()
    parent = dirname(p)
    if base in ("samples", "sample") and isdir(parent) and _dir_has_images(parent):
        return parent

    for sub in ("samples", "Samples", "SAMPLES", "sample", "Sample"):
        sp = join(p, sub)
        if isdir(sp) and _dir_has_images(sp):
            return sp

    subs = [d for d in listdir(p) if isdir(join(p, d))]
    if len(subs) == 1:
        return join(p, subs[0])

    return p


def _abs_to_assets_url(p: str) -> str:
    if not p:
        return p

    p_norm = p.replace("\\", "/")
    root = ASSETS_FOLDER.replace("\\", "/").rstrip("/")

    if p_norm.startswith(root + "/"):
        rel = p_norm[len(root) + 1:]
        rel_enc = "/".join(quote(seg) for seg in rel.split("/"))
        return f"{ASSETS_URL_PATH}/{rel_enc}"

    if p_norm.startswith("assets/"):
        rel = p_norm[len("assets/"):]
        rel_enc = "/".join(quote(seg) for seg in rel.split("/"))
        return f"{ASSETS_URL_PATH}/{rel_enc}"

    if not p_norm.startswith("/"):
        rel_enc = "/".join(quote(seg) for seg in p_norm.split("/"))
        return f"{ASSETS_URL_PATH}/{rel_enc}"

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


def _is_checked_list(val, label_text: str) -> bool:
    return isinstance(val, list) and (label_text in val)


def _df_from_store(s_store_df: str) -> pd.DataFrame:
    try:
        return pd.read_json(StringIO(s_store_df))
    except Exception:
        return pd.DataFrame()


def _cache_get_batch(csv_basename: str) -> pd.DataFrame | None:
    dfc = BATCH_DF_CACHE.get(csv_basename)
    return dfc if isinstance(dfc, pd.DataFrame) else None


def _cache_set_batch(bn: str, df: pd.DataFrame) -> None:
    with _CACHE_LOCK:
        BATCH_DF_CACHE[bn] = df.copy()


# =========================
# CLUSTER ARTIFACT HELPERS (NOVO)
# =========================
def _repo_root_guess() -> Path:
    here = Path(__file__).resolve()
    p = here.parent
    for _ in range(10):
        if (p / "main" / "assets").is_dir():
            return p
        p = p.parent
    return here.parent


def _project_dir(project: str) -> Path:
    root = _repo_root_guess()
    return root / "main" / "assets" / str(project)


def _sanitize_cluster_filename(x: str) -> str:
    x = str(x).replace("/", "_").replace("\\", "_").strip()
    x = re.sub(r"\s+", "_", x)
    return x


def _cluster_npz_path(project: str, class_label: str) -> Path:
    pdir = _project_dir(project)
    return pdir / CLUSTERS_SUBDIR / f"{_sanitize_cluster_filename(class_label)}.npz"


def _load_cluster_artifact(project: str, class_label: str) -> dict | None:
    npz_path = _cluster_npz_path(project, class_label)
    if not npz_path.exists():
        return None

    key = (str(project), _sanitize_cluster_filename(class_label))
    mtime = float(npz_path.stat().st_mtime)

    cached = CLUSTER_ARTIFACT_CACHE.get(key)
    if cached and cached.get("mtime") == mtime:
        return cached["data"]

    z = np.load(npz_path, allow_pickle=True)
    files = set(z.files)

    if "names" not in files or "linkage" not in files:
        return None

    names = z["names"].astype(object)
    Z = z["linkage"].astype(np.float64)
    dist_max = float(z["dist_max"]) if "dist_max" in files else 1.0

    data = {
        "path": str(npz_path),
        "names": names,
        "Z": Z,
        "dist_max": dist_max,
        "mode": str(z["mode"]) if "mode" in files else "",
        "metric": str(z["metric"]) if "metric" in files else "",
        "method": str(z["method"]) if "method" in files else "",
        "pca_dim": int(z["pca_dim"]) if "pca_dim" in files else None,
    }
    CLUSTER_ARTIFACT_CACHE[key] = {"mtime": mtime, "data": data}
    return data


def _run_class_clustering(project: str, one_class: str | None = None) -> tuple[bool, str]:
    script = _repo_root_guess() / "class_clustering.py"
    if not script.exists():
        return False, f"Script não encontrado: {script}"

    cmd = [sys.executable, str(script), str(project)]
    if one_class:
        cmd += ["--class", str(one_class)]
        
    cmd += ["--metric", "cosine", "--linkage", "average", "--pca-dim", "64"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = (r.returncode == 0)
        out = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
        return ok, out[-8000:]
    except Exception as e:
        return False, f"Falha ao executar: {type(e).__name__}: {e}"


# =========================
# PATH / BATCH HELPERS
# =========================
def _infer_batch_folder_from_csv_basename(csv_basename_no_ext: str):
    m = re.match(r"^(batch_\d+)", csv_basename_no_ext)
    return m.group(1) if m else None


def _replace_batch_in_path(template_path: str, new_batch_folder: str) -> str:
    parts = [p for p in template_path.replace("\\", "/").split("/") if p]
    for i, p in enumerate(parts):
        if re.match(r"^batch_\d+$", p):
            parts[i] = new_batch_folder
            return ("/" + "/".join(parts)) if template_path.startswith("/") else "/".join(parts)
    return template_path


def _resolve_background(project_name: str, batch_str: str):
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
# CSV COLUMN / MINIMAL READ HELPERS
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
    if csv_basename in BATCH_DF_CACHE:
        dfc = _cache_get_batch(csv_basename)
        if isinstance(dfc, pd.DataFrame):
            return dfc.copy()
        # fallback ultra-seguro
        dfc = BATCH_DF_CACHE.get(csv_basename)
        return dfc.copy() if isinstance(dfc, pd.DataFrame) else pd.DataFrame()


    cols = _get_csv_columns(csv_path, csv_basename)
    want = ["names", "custom_data", "manual_label", "correct_label", "x", "y", "confs1", "var_norm", "colors", "embed_file", "embed_idx"]
    usecols = [c for c in want if c in cols]

    if not usecols:
        df = pd.read_csv(csv_path, encoding="ISO-8859-1", low_memory=False)
    else:
        df = pd.read_csv(csv_path, encoding="ISO-8859-1", usecols=usecols, low_memory=False)

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

    for c in ("x", "y", "confs1", "var_norm"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _compute_batch_bbox(csv_path: str, csv_basename: str) -> tuple[float, float, float, float] | None:
    if csv_basename in BATCH_BBOX_CACHE:
        return BATCH_BBOX_CACHE[csv_basename]

    try:
        if csv_basename in BATCH_DF_CACHE:
            df = _cache_get_batch(csv_basename)
            if isinstance(df, pd.DataFrame):
                df = df.copy()
            else:
                df = pd.DataFrame()

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
    if poly_xy is None or len(poly_xy) < 3:
        return np.zeros_like(x, dtype=bool)

    poly = [(float(px), float(py)) for px, py in poly_xy]
    if poly[0] != poly[-1]:
        poly.append(poly[0])

    px = np.array([p[0] for p in poly], dtype=np.float64)
    py = np.array([p[1] for p in poly], dtype=np.float64)

    n = len(poly) - 1
    inside = np.zeros(x.shape[0], dtype=bool)

    eps = 1e-12
    for i in range(n):
        x1, y1 = px[i], py[i]
        x2, y2 = px[i + 1], py[i + 1]
        cond = ((y1 > y) != (y2 > y))
        x_int = (x2 - x1) * (y - y1) / (y2 - y1 + eps) + x1
        inside ^= (cond & (x < x_int))
    return inside


def _parse_selection_polygon(selectedData: dict | None) -> dict | None:
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

    df["names"] = df["names"].astype(str)
    df["custom_data"] = df["custom_data"].astype(str)
    df["manual_label"] = df["manual_label"].astype(str)
    df["correct_label"] = df["correct_label"].astype(str)

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
    if csv_basename in BATCH_DF_CACHE:
        dfc = _cache_get_batch(csv_basename)
        if isinstance(dfc, pd.DataFrame):
            return dfc.copy()
        dfc = BATCH_DF_CACHE.get(csv_basename)
        return dfc.copy() if isinstance(dfc, pd.DataFrame) else pd.DataFrame()


    df0 = _ensure_required_columns(read_input_csv(csv_path))
    _cache_set_batch(csv_basename, df0)
    return df0.copy()


# =========================
# OVERVIEW (all batches)
# =========================
def _palette_rgb(i: int):
    pal = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),

        (174, 199, 232),
        (255, 187, 120),
        (152, 223, 138),
        (255, 152, 150),
        (197, 176, 213),
        (196, 156, 148),
        (247, 182, 210),
        (199, 199, 199),
        (219, 219, 141),
        (158, 218, 229),

        (214, 97, 107),
        (86, 180, 233),
        (230, 171, 2),
        (0, 158, 115),
        (240, 228, 66),
        (0, 114, 178),
        (213, 94, 0),
        (204, 121, 167),
    ]
    return pal[i % len(pal)]


def _gather_overview_points(project_name: str, csv_folder: str):
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
    color_mapping = {}
    csvs = _list_csv_candidates(csv_folder)

    for csv_file in csvs:
        if csv_file == OVERVIEW_VALUE:
            continue

        csv_path = join(csv_folder, csv_file)
        try:
            df_sample = pd.read_csv(csv_path, usecols=['manual_label', 'colors'], nrows=5000, encoding="ISO-8859-1")

            for label in df_sample['manual_label'].astype(str).unique():
                if label in labels and label not in color_mapping:
                    color_id = int(df_sample.loc[df_sample['manual_label'].astype(str) == label, 'colors'].iloc[0])
                    color_mapping[label] = color_id

            if all(label in color_mapping for label in labels):
                break

        except Exception:
            continue

    for i, label in enumerate(labels):
        if label not in color_mapping:
            color_mapping[label] = i

    return color_mapping


def _build_overview_figure(project_name: str, csv_folder: str, graph_range: int) -> go.Figure:
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
            groupclick="toggleitem",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        title="Overview (all batches): hull + densidade por classe",
        dragmode="lasso",
    )

    labels_sorted = sorted(pts_by_label.keys(), key=lambda s: s.lower())
    color_mapping = _get_class_colors_from_batches(csv_folder, labels_sorted)

    for i, lab in enumerate(labels_sorted):
        xy = pts_by_label.get(lab)
        if xy is None or xy.size == 0:
            continue

        x = xy[:, 0]
        y = xy[:, 1]

        color_id = color_mapping.get(lab, i)
        r, g, b = _palette_rgb(color_id)

        c0 = f"rgba({r},{g},{b},0.00)"
        c1 = f"rgba({r},{g},{b},0.35)"
        line_col = f"rgba({r},{g},{b},0.95)"  # <<< FIX (4): mesma cor do heat

        # Heat hull (densidade)
        fig.add_trace(
            go.Histogram2dContour(
                x=x,
                y=y,
                contours=dict(coloring="fill", showlines=False),
                ncontours=10,
                showscale=False,
                colorscale=[(0.0, c0), (1.0, c1)],
                hoverinfo="skip",
                name=lab,
                showlegend=False,
                legendgroup=lab,
                opacity=1.0,
            )
        )

        # Convex hull / bbox (linha agora com cor explícita)
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
                        line=dict(width=1.5, color=line_col),  # <<< FIX (4)
                        fill="toself",
                        fillcolor=f"rgba({r},{g},{b},0.05)",
                        name=lab,
                        hoverinfo="skip",
                        showlegend=True,
                        legendgroup=lab,
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
                        line=dict(width=1.5, color=line_col),  # <<< FIX (4)
                        fill="toself",
                        fillcolor=f"rgba({r},{g},{b},0.04)",
                        name=lab,
                        hoverinfo="skip",
                        showlegend=True,
                        legendgroup=lab,
                        visible=True,
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[], y=[],
                    mode="lines",
                    line=dict(width=1.5, color=line_col),
                    name=lab,
                    showlegend=True,
                    legendgroup=lab,
                )
            )

    # camada invisível para seleção
    rng = np.random.default_rng(123)
    for lab in labels_sorted:
        xy = pts_by_label.get(lab)
        if xy is None or xy.size == 0:
            continue
        k = min(1500, xy.shape[0])
        if k <= 0:
            continue
        idx = rng.choice(xy.shape[0], size=k, replace=False)
        sel = xy[idx]

        fig.add_trace(
            go.Scatter(
                x=sel[:, 0],
                y=sel[:, 1],
                mode="markers",
                marker=dict(size=6, color="rgba(0,0,0,0.003)"),
                hoverinfo="skip",
                name=lab,
                showlegend=False,
                legendgroup=lab,
                selected=dict(marker=dict(color="rgba(0,0,0,0.003)")),
                unselected=dict(marker=dict(color="rgba(0,0,0,0.003)")),
                customdata=[[lab] for _ in range(k)],
                opacity=0.001,
                visible=True,
            )
        )

    fig.update_layout(legend=dict(groupclick="togglegroup"))
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

# app.py  (PARTE 2/5)  — ALLB + DENDRO TRUNCADO + HELPERS CLUSTER (.npz)

# =========================
# ALL-BATCHES selection (máscaras compactas + paginação determinística)
# =========================
def _mask_for_df_by_spec(dfb: pd.DataFrame, spec: dict) -> np.ndarray:
    """
    Retorna mask boolean (len==len(dfb)) para os filtros do all-batches selection.
    Suporta class_filter = None | str | list[str].
    """
    n = int(len(dfb))
    if n <= 0:
        return np.zeros((0,), dtype=bool)

    if "x" not in dfb.columns or "y" not in dfb.columns:
        return np.zeros((n,), dtype=bool)

    x = pd.to_numeric(dfb["x"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(dfb["y"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)

    geom = spec.get("geom", {})
    gtype = geom.get("type")

    if gtype == "lasso":
        poly = geom.get("poly", [])
        ok &= _points_in_poly(x, y, poly)
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

    cls = spec.get("class_filter", None)
    if cls:
        if "manual_label" in dfb.columns:
            labs = dfb["manual_label"].astype(str).to_numpy(dtype=object)
            if isinstance(cls, str):
                ok &= (labs == str(cls))
            elif isinstance(cls, list) and len(cls) > 0:
                ok &= np.isin(labs, [str(c) for c in cls])
            elif isinstance(cls, list) and len(cls) == 0:
                ok &= False
        else:
            return np.zeros((n,), dtype=bool)

    pr = spec.get("probs_rng", None)
    if pr is not None and "confs1" in dfb.columns:
        c1 = pd.to_numeric(dfb["confs1"], errors="coerce").to_numpy(dtype=np.float64)
        ok &= np.isfinite(c1) & (c1 >= float(pr[0])) & (c1 <= float(pr[1]))

    vr = spec.get("vars_rng", None)
    if vr is not None and "var_norm" in dfb.columns:
        v1 = pd.to_numeric(dfb["var_norm"], errors="coerce").to_numpy(dtype=np.float64)
        ok &= np.isfinite(v1) & (v1 >= float(vr[0])) & (v1 <= float(vr[1]))

    if bool(spec.get("hide_relab", False)):
        if "manual_label" in dfb.columns and "correct_label" in dfb.columns:
            ml = dfb["manual_label"].astype(str).to_numpy(dtype=object)
            cl = dfb["correct_label"].astype(str).to_numpy(dtype=object)
            ok &= (ml == cl)

    return ok.astype(bool, copy=False)


def _unchecked_by_batch(unchecked_set: set[str]) -> dict[str, set[str]]:
    """
    unchecked_set contém global_id: "batch.csv::custom_data_orig"
    Retorna {batch.csv: {custom_data_orig,...}}
    """
    out: dict[str, set[str]] = {}
    for gid in unchecked_set:
        sp = _split_global_id(gid)
        if not sp:
            continue
        bn, cd_orig = sp
        out.setdefault(bn, set()).add(cd_orig)
    return out


def _get_visible_classes_from_figure(fig_dict: dict) -> list[str]:
    if not fig_dict or "data" not in fig_dict:
        return []
    vis = []
    for tr in fig_dict["data"]:
        if tr.get("showlegend", False):
            nm = tr.get("name", "")
            v = tr.get("visible", True)
            if nm and v not in ("legendonly", False):
                vis.append(str(nm))
    # únicos
    out = []
    seen = set()
    for x in vis:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _allb_build_selection(
    project_name: str,
    csv_folder: str,
    csv_candidates: list[str],
    spec: dict,
    version: int,
    visible_classes: list | None = None
) -> tuple[str, int]:
    """
    Constrói máscaras por batch (compactadas) e prefix-sums para paginação determinística.
    """
    batch_order = [f for f in csv_candidates if f != OVERVIEW_VALUE]
    token = _new_token()

    counts = [0] * len(batch_order)
    cum_counts = [0] * len(batch_order)
    packmasks: dict[str, bytes] = {}
    lens: dict[str, int] = {}
    index_map = {bn: i for i, bn in enumerate(batch_order)}

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
        idx = index_map[bn]

        if not exists(p):
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
        counts[idx] = c

        if c > 0:
            pack = np.packbits(mask.astype(np.uint8))
            packmasks[bn] = pack.tobytes()
        else:
            packmasks.pop(bn, None)

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


def _read_batch_full_for_page(csv_path: str, csv_basename: str) -> pd.DataFrame:
    """
    Para montar página: se batch está no cache (modificado/aberto), usa ele; senão lê do disco (full).
    """
    # 1) cache principal (batches modificados/abertos)
    if csv_basename in BATCH_DF_CACHE:
        dfc = BATCH_DF_CACHE[csv_basename]
        return dfc.copy() if isinstance(dfc, pd.DataFrame) else pd.DataFrame()

    # 2) (opcional) cache getter se existir no seu código
    #    NUNCA use `df or ...` com DataFrame
    try:
        dfc = _cache_get_batch(csv_basename)  # pode retornar DataFrame ou None
        if isinstance(dfc, pd.DataFrame):
            # se você quiser aceitar DF vazio aqui, mantenha como está; eu prefiro retornar mesmo vazio
            return dfc.copy()
    except Exception:
        pass

    # 3) fallback: lê do disco
    try:
        df = _ensure_required_columns(read_input_csv(csv_path))
        return df
    except Exception:
        return pd.DataFrame()



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

    start_b = bisect_right(cum, start)
    end_b = bisect_right(cum, end - 1)

    rows_out = []

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

        # widths/heights só para itens da página
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
    for c in ["names", "custom_data", "manual_label", "correct_label", "csv_basename", "custom_data_orig"]:
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out


def _allb_collect_selected_for_class(
    token: str,
    s_ctx: dict,
    class_label: str,
    unchecked_set: set[str] | None = None
) -> pd.DataFrame:
    """
    Retorna DF com TODOS os pontos do token (todas as páginas) já filtrando por class_label,
    e removendo unchecked_set (global_ids).
    """
    cache = ALLB_CACHE.get(token)
    if not cache:
        return pd.DataFrame()

    _csv_folder = str(s_ctx.get("csv_folder", ""))
    batch_order = cache.get("batch_order", [])
    packmasks = cache.get("packmasks", {})
    lens = cache.get("lens", {})

    unchecked_set = unchecked_set or set()
    unchecked_by_bn = _unchecked_by_batch(unchecked_set)

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

        # remove unchecked desse batch
        bad_cd = unchecked_by_bn.get(bn, set())
        if bad_cd:
            part = part[~part["custom_data"].astype(str).isin(bad_cd)].copy()
            if part.empty:
                continue

        part["csv_basename"] = bn
        part["custom_data_orig"] = part["custom_data"].astype(str)
        part["custom_data"] = [_global_id(bn, cd) for cd in part["custom_data_orig"].astype(str).tolist()]
        rows.append(part)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    for c in ["names", "custom_data", "custom_data_orig", "manual_label", "correct_label", "csv_basename"]:
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out


# =========================
# DENDROGRAM (TRUNCADO P/ CLASSES GRANDES)  — FIX (1)
# =========================
def _build_dendrogram_figure(Z: np.ndarray, cut_dist: float) -> go.Figure:
    fig = go.Figure()
    if dendrogram is None:
        fig.add_annotation(text="scipy não disponível: não é possível desenhar dendrograma.", showarrow=False)
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    if Z is None or (not hasattr(Z, "size")) or (Z.size == 0):
        fig.add_annotation(text="Sem dados suficientes para dendrograma.", showarrow=False)
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    Z = np.asarray(Z, dtype=np.float64)
    n = Z.shape[0] + 1
    hard_cap = 50000  # igual ao data_explorer (ajuste se quiser)

    if n > hard_cap:
        fig.add_annotation(
            text=f"Dendrograma não desenhado (N={n} > hard_cap={hard_cap}).",
            showarrow=False
        )
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
        return fig

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

    mpl_cycle = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                 "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

    def _map_mpl_color(c: str) -> str:
        if isinstance(c, str) and len(c) == 2 and c[0] == "C" and c[1].isdigit():
            return mpl_cycle[int(c[1]) % len(mpl_cycle)]
        return c if isinstance(c, str) else "#1f77b4"

    traces = {}
    for xs, ys, col in zip(icoord, dcoord, color_list):
        is_above = (max(ys) >= cut_dist)
        key_color = "black" if is_above else _map_mpl_color(col)
        traces.setdefault(key_color, {"x": [], "y": []})
        traces[key_color]["x"].extend([xs[0], xs[1], xs[2], xs[3], None])
        traces[key_color]["y"].extend([ys[0], ys[1], ys[2], ys[3], None])

    for c, xy in traces.items():
        fig.add_trace(go.Scatter(
            x=xy["x"], y=xy["y"], mode="lines",
            line=dict(color=c, width=1.5),
            hoverinfo="skip", showlegend=False
        ))

    fig.add_hline(y=cut_dist, line_dash="dash", line_color="black", opacity=0.9)
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title="distance", showgrid=False, zeroline=False),
        title="Dendrograma",
    )
    return fig


# Coloque isso em algum lugar perto das configs do app.py
CLUSTER_CUT_MODE = "relative"  # "relative" (0..1 vira % do dist_max) ou "absolute" (0..1 é distância bruta)

def _npz_scalar(v, default=None):
    if v is None:
        return default
    try:
        if isinstance(v, np.ndarray):
            if v.shape == ():
                return v.item()
            if v.size == 1:
                return v.reshape(()).item()
        return v
    except Exception:
        return default


def _cluster_ids_from_artifact(project: str, class_label: str, cut01: float) -> tuple[dict[str, int], dict]:
    """
    Usa clusters/<class>.npz:
      - retorna name->cluster_id (1..K)
      - meta: dist_max, cut_dist, N_total, groups_total, artifact_path, cut_mode
    Se não existir, tenta gerar via class_clustering.py.
    """
    cut01 = float(cut01) if cut01 is not None else 0.50
    cut01 = min(max(cut01, 0.0), 1.0)

    art = _load_cluster_artifact(project, class_label)
    if art is None:
        ok, out = _run_class_clustering(project, one_class=class_label)
        if not ok:
            return {}, {"error": f"Falha ao gerar .npz para '{class_label}'. Log:\n{out}"}
        art = _load_cluster_artifact(project, class_label)
        if art is None:
            return {}, {"error": f"Não consegui carregar o .npz mesmo após gerar (classe='{class_label}')."}

    if fcluster is None:
        return {}, {"error": "scipy fcluster não disponível. Instale scipy."}

    Z = np.asarray(art["Z"])
    names = np.asarray(art["names"], dtype=object)

    # dist_max SEMPRE a partir do Z real (evita drift se dist_max salvo estiver arredondado)
    if Z.size and Z.shape[0] > 0:
        dist_max = float(np.nanmax(Z[:, 2]))
        if not np.isfinite(dist_max) or dist_max <= 0:
            dist_max = 1.0
    else:
        dist_max = float(_npz_scalar(art.get("dist_max", 1.0), 1.0)) or 1.0

    cut_mode = (CLUSTER_CUT_MODE or "relative").lower().strip()
    if cut_mode not in ("relative", "absolute"):
        cut_mode = "relative"

    if cut_mode == "absolute":
        cut_dist = float(cut01)
    else:
        cut_dist = float(cut01) * float(dist_max)

    try:
        # fcluster trabalha melhor com float64 (especialmente perto do threshold)
        Z64 = Z.astype(np.float64, copy=False)
        cids = fcluster(Z64, t=cut_dist, criterion="distance")
        cids = np.asarray(cids).astype(int)
    except Exception as e:
        return {}, {"error": f"fcluster falhou: {type(e).__name__}: {e}"}

    name2cid = {str(nm): int(cid) for nm, cid in zip(names.tolist(), cids.tolist())}
    K = int(np.max(cids)) if cids.size else 0

    meta = {
        "artifact_path": art.get("path", ""),
        "N_total": int(len(names)),
        "groups_total": int(K),
        "dist_max": float(dist_max),
        "cut01": float(cut01),
        "cut_dist": float(cut_dist),
        "cut_mode": cut_mode,
        "source": "artifact",
        "mode": str(_npz_scalar(art.get("mode", ""), "")),
        "metric": str(_npz_scalar(art.get("metric", ""), "")),
        "method": str(_npz_scalar(art.get("method", ""), "")),
        "pca_dim": int(_npz_scalar(art.get("pca_dim", 0), 0)),
        "max_full": int(_npz_scalar(art.get("max_full", 0), 0)),
    }
    return name2cid, meta


# =========================
# SELECTION-BASED CLUSTERING (para replicar comportamento "old")
# =========================
_EMBED_FEATS_CACHE: dict[str, np.ndarray] = {}

def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n

def _resolve_embed_path_from_project(project: str, embed_file_value: str) -> Path:
    """
    embed_file pode vir:
      - só nome (batch_0001.npz)
      - relativo (embeddings/batch_0001.npz)
      - absoluto
    """
    project_dir = _project_dir(project)
    p = Path(str(embed_file_value))

    if p.is_absolute() and p.exists():
        return p

    cand1 = project_dir / p
    if cand1.exists():
        return cand1

    cand2 = project_dir / "embeddings" / p.name
    if cand2.exists():
        return cand2

    if "embeddings" in p.parts:
        tail = Path(*p.parts[p.parts.index("embeddings") + 1 :])
        cand3 = project_dir / "embeddings" / tail
        if cand3.exists():
            return cand3

    raise FileNotFoundError(f"embed_file não encontrado: '{embed_file_value}' (project={project_dir})")

def _load_feats_from_npz(npz_path: Path) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=False)
    for k in ("feats", "embeddings", "vectors", "X"):
        if k in z.files:
            arr = z[k]
            break
    else:
        arr = None
        for k in z.files:
            a = z[k]
            if isinstance(a, np.ndarray) and a.ndim == 2:
                arr = a
                break
        if arr is None:
            raise KeyError(f"{npz_path.name}: não achei array 2D de embeddings (keys={list(z.files)})")

    if arr.ndim != 2:
        raise ValueError(f"{npz_path.name}: embeddings não são 2D (shape={arr.shape})")

    return arr

def _gather_X_for_df_sel(project: str, df_sel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Monta X (N,D) e names (N,) a partir do df_sel.
    Precisa de colunas: names, embed_file, embed_idx.
    """
    if df_sel is None or df_sel.empty:
        return np.empty((0, 0), np.float32), np.array([], dtype=object)

    if "names" not in df_sel.columns:
        return np.empty((0, 0), np.float32), np.array([], dtype=object)

    if ("embed_file" not in df_sel.columns) or ("embed_idx" not in df_sel.columns):
        # Sem embed cols => não dá pra replicar o "old" com embeddings
        raise RuntimeError("df_sel não tem embed_file/embed_idx. Ajuste _read_batch_minimal para incluir.")

    t = df_sel[["names", "embed_file", "embed_idx"]].copy()
    t["names"] = t["names"].astype(str)
    t["embed_file"] = t["embed_file"].astype(str)
    t["embed_idx"] = pd.to_numeric(t["embed_idx"], errors="coerce").fillna(-1).astype(np.int64)
    t = t[(t["embed_idx"] >= 0) & t["names"].notna() & t["embed_file"].notna()]
    if t.empty:
        return np.empty((0, 0), np.float32), np.array([], dtype=object)

    # ordem: exatamente a ordem do df_sel filtrado
    names = t["names"].to_numpy(dtype=object)
    embed_files = t["embed_file"].to_numpy(dtype=object)
    embed_idxs = t["embed_idx"].to_numpy(dtype=np.int64)

    # descobre D
    first_path = _resolve_embed_path_from_project(project, str(embed_files[0]))
    k0 = str(first_path)
    if k0 not in _EMBED_FEATS_CACHE:
        _EMBED_FEATS_CACHE[k0] = _load_feats_from_npz(first_path)
    D = int(_EMBED_FEATS_CACHE[k0].shape[1])

    X = np.empty((len(names), D), dtype=np.float32)

    file_to_pos: dict[str, list[int]] = {}
    for pos, ef in enumerate(embed_files):
        file_to_pos.setdefault(str(ef), []).append(pos)

    for ef, positions in file_to_pos.items():
        ep = _resolve_embed_path_from_project(project, ef)
        kk = str(ep)
        if kk not in _EMBED_FEATS_CACHE:
            _EMBED_FEATS_CACHE[kk] = _load_feats_from_npz(ep)

        feats = _EMBED_FEATS_CACHE[kk]
        idxs = embed_idxs[np.array(positions, dtype=np.int64)]
        ok = (idxs >= 0) & (idxs < feats.shape[0])
        pos_ok = np.array(positions, dtype=np.int64)[ok]
        idx_ok = idxs[ok]
        if len(pos_ok):
            X[pos_ok, :] = feats[idx_ok, :].astype(np.float32, copy=False)

    return X, names

def _cluster_ids_from_selection(project: str, class_label: str, cut01: float, df_sel: pd.DataFrame,
                                metric: str = "cosine", method: str = "average", pca_dim: int = 64) -> tuple[dict[str, int], dict]:
    """
    Clusteriza SOMENTE o df_sel (com embeddings), para replicar o comportamento antigo.
    """
    if linkage is None or fcluster is None:
        return {}, {"error": "scipy linkage/fcluster não disponível. Instale scipy.", "source": "selection"}

    cut01 = float(cut01) if cut01 is not None else 0.50
    cut01 = min(max(cut01, 0.0), 1.0)

    if df_sel is None or df_sel.empty:
        return {}, {"error": "Seleção vazia.", "source": "selection"}

    N_sel = int(len(df_sel))
    if N_sel < 2:
        return {}, {"error": "Seleção pequena demais (N<2).", "source": "selection"}

    if N_sel > MAX_CLUSTER_N:
        return {}, {"error": f"Seleção grande demais p/ clustering local (N={N_sel} > MAX_CLUSTER_N={MAX_CLUSTER_N}).", "source": "selection"}

    X, names = _gather_X_for_df_sel(project, df_sel)
    if X.size == 0 or len(names) < 2:
        return {}, {"error": "Não consegui montar embeddings da seleção.", "source": "selection"}

    # pipeline: L2 -> PCA(opc) -> L2
    X = _l2_normalize(X)

    if pca_dim and pca_dim > 0 and PCA is not None:
        ncomp = int(min(int(pca_dim), X.shape[0], X.shape[1]))
        if 2 <= ncomp < X.shape[1]:
            pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=0)
            X = pca.fit_transform(X).astype(np.float32, copy=False)
            X = _l2_normalize(X)

    # linkage e corte
    Z = linkage(X.astype(np.float64, copy=False), method=method, metric=metric)
    dist_max = float(np.nanmax(Z[:, 2])) if Z.size else 1.0
    if (not np.isfinite(dist_max)) or dist_max <= 0:
        dist_max = 1.0

    cut_mode = (CLUSTER_CUT_MODE or "relative").lower().strip()
    if cut_mode not in ("relative", "absolute"):
        cut_mode = "relative"

    cut_dist = float(cut01) if cut_mode == "absolute" else float(cut01) * float(dist_max)

    cids = fcluster(Z.astype(np.float64, copy=False), t=cut_dist, criterion="distance").astype(int)

    name2cid = {str(nm): int(cid) for nm, cid in zip(names.tolist(), cids.tolist())}
    K = int(np.max(cids)) if cids.size else 0

    meta = {
        "source": "selection",
        "N_total": int(len(names)),
        "groups_total": int(K),
        "dist_max": float(dist_max),
        "cut01": float(cut01),
        "cut_dist": float(cut_dist),
        "cut_mode": cut_mode,
        "metric": metric,
        "method": method,
        "pca_dim": int(pca_dim),
        "Z": Z,  # dendrograma local
    }
    return name2cid, meta

def _cluster_ids_best(project: str, class_label: str, cut01: float, df_sel: pd.DataFrame | None) -> tuple[dict[str, int], dict]:
    """
    Tenta selection-clustering (old). Se falhar/ficar grande, cai no artifact.
    """
    if df_sel is not None and isinstance(df_sel, pd.DataFrame) and (len(df_sel) >= 2) and (len(df_sel) <= MAX_CLUSTER_N):
        n2c, meta = _cluster_ids_from_selection(project, class_label, cut01, df_sel,
                                               metric="cosine", method="average", pca_dim=64)
        if "error" not in meta:
            return n2c, meta

    # fallback
    return _cluster_ids_from_artifact(project, class_label, cut01)


# app.py  (PARTE 3/5) — INIT + UI (botão Update cluster files + alert) + helpers de “dirty classes”

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
    csv_basename=csv_basename,
    csv_file=csv_file,
    path_to_images=path_to_images,
    path_to_thumbnails=path_to_images,
    background_path=background_path,
    thumbs_jpg=thumbs_jpg,
    has_var=HAS_VAR,
    images_path_template=path_to_images,
    is_overview=True,
    overview_version=0,
    allb_active=False,
)

# =========================
# DIRTY CLASSES (p/ update clusters artifacts) — FIX (5)
# =========================
DIRTY_CLASSES: set[str] = set()

def _mark_dirty_class(*labs: str):
    for x in labs:
        if x is None:
            continue
        s = str(x).strip()
        if s:
            DIRTY_CLASSES.add(s)

def _mark_dirty_from_changes(old_labels: list[str], new_labels: list[str]):
    for x in old_labels:
        _mark_dirty_class(x)
    for x in new_labels:
        _mark_dirty_class(x)

def _project_dir_from_ctx(s_ctx: dict) -> str:
    proj = str((s_ctx or {}).get("project_name", project_name))
    return join(ASSETS_FOLDER, proj)

def _count_classes_for_targets(csv_folder_path: str, targets: set[str]) -> dict[str, int]:
    """
    Conta apenas as classes em `targets` varrendo todos os CSVs.
    Leitura leve: só manual_label/correct_label fallback.
    """
    if not targets:
        return {}
    targets = set(str(x) for x in targets)

    counts = {t: 0 for t in targets}
    csvs = _list_csv_candidates(csv_folder_path)

    for bn in csvs:
        if bn == OVERVIEW_VALUE:
            continue
        p = join(csv_folder_path, bn)
        if not exists(p):
            continue

        try:
            cols = _get_csv_columns(p, bn)
            if "manual_label" in cols:
                usecols = ["manual_label"]
                dfc = pd.read_csv(p, encoding="ISO-8859-1", usecols=usecols, low_memory=False)
                s = dfc["manual_label"].astype(str)
            elif "correct_label" in cols:
                usecols = ["correct_label"]
                dfc = pd.read_csv(p, encoding="ISO-8859-1", usecols=usecols, low_memory=False)
                s = dfc["correct_label"].astype(str)
            else:
                continue

            vc = s.value_counts(dropna=False)
            for t in targets:
                if t in vc.index:
                    counts[t] += int(vc.loc[t])
        except Exception:
            continue

    return counts

def _artifact_count(project: str, class_label: str) -> int | None:
    art = _load_cluster_artifact(project, class_label)
    if not art:
        return None
    try:
        return int(len(art.get("names", [])))
    except Exception:
        return None


def _delete_cluster_artifact(project: str, class_label: str) -> tuple[bool, str]:
    """
    Deleta clusters/<class>.npz (se existir) e limpa o cache.
    Retorna (ok, msg).
    """
    npz_path = _cluster_npz_path(project, class_label)
    if not npz_path.exists():
        # já não existe -> ok (nada a remover)
        key = (str(project), _sanitize_cluster_filename(class_label))
        CLUSTER_ARTIFACT_CACHE.pop(key, None)
        return True, "missing"

    try:
        npz_path.unlink()
        key = (str(project), _sanitize_cluster_filename(class_label))
        CLUSTER_ARTIFACT_CACHE.pop(key, None)
        return True, str(npz_path)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


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
            html.Div("Multi-Batch Selection", style={"fontWeight": "600","marginBottom": "10px","fontSize": "14px","color": "#333"}),
            dbc.Row([dbc.Col(html.Div("Classes: mesmo filtro da legenda do plot", style={"fontSize": "12px","marginBottom": "2px","color": "#666","fontStyle": "italic"}), width=12)]),
            dbc.Row([dbc.Col(html.Div("Page Size", style={"fontSize": "12px","marginBottom": "2px"}), width=12)]),
            dbc.Row(
                [dbc.Col(dcc.Input(id="input_allb_page_size", type="number", min=100, max=MAX_ALLB_PAGE_SIZE, step=100, value=2000,
                                   style={"width": "100%","fontSize": "12px","padding": "6px"}), width=12)],
                align="center", className="mb-2",
            ),
            dbc.Row([dbc.Col(dbc.Button("Build Selection", id="button_allb_build", n_clicks=0, color="primary", size="sm", style={"width": "100%"}), width=12)], className="mb-2"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.ButtonGroup(
                            [
                                dbc.Button("◀ Prev", id="button_allb_prev", n_clicks=0, color="secondary", size="sm"),
                                dbc.Button("Next ▶", id="button_allb_next", n_clicks=0, color="secondary", size="sm"),
                            ],
                            style={"width": "100%"},
                        ),
                        width=8,
                    ),
                    dbc.Col(
                        html.Div(
                            id="allb_page_info",
                            style={"fontFamily": "monospace","fontSize": "11px","padding": "5px","backgroundColor": "#f8f9fa","borderRadius": "3px","border": "1px solid #dee2e6"},
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
            html.Div("Labeling", style={"fontWeight": "600","marginBottom": "10px","fontSize": "14px","color": "#333"}),
            dbc.Row(
                [
                    dbc.Col(dbc.Input(placeholder="Enter new label", id="input_aplicar_novo_label", type="text", style={"fontSize": "12px"}), width=8),
                    dbc.Col(dbc.Button("Apply", id="button_aplicar_novo_label", color="primary", size="sm", style={"width": "100%"}), width=4),
                ],
                className="mb-2",
            ),
            dbc.Row([dbc.Col(dbc.Checklist(options=[{"label": " Save after labeling", "value": " Save CSV after labeling"}],
                                          value=[" Save CSV after labeling"], id="check_save_csv", inline=True, style={"fontSize": "12px"}), width=12)], className="mb-2"),
            dbc.Row([dbc.Col(dbc.Checklist(options=[{"label": " Discard after labeling", "value": " Discard images after labeling"}],
                                          value=[" Discard images after labeling"], id="check_discard_marked", inline=True, style={"fontSize": "12px"}), width=12)]),
        ]
    ),
    className="mb-3",
)

# --- Image Display Controls ---
display_card = dbc.Card(
    dbc.CardBody(
        [
            html.Div("Display Options", style={"fontWeight": "600","marginBottom": "10px","fontSize": "14px","color": "#333"}),
            dbc.Row(
                [
                    dbc.Col(html.Div("Marker Size", style={"fontSize": "12px","marginBottom": "5px"}), width=12),
                    dbc.Col(dcc.Slider(1, 25, 3, value=10, id="slider_marker_size", marks=None, tooltip={"placement": "bottom", "always_visible": True}), width=12),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div("Label Order", style={"fontSize": "12px","marginBottom": "2px"}), width=12),
                    dbc.Col(dcc.Dropdown(["A-Z, a-z", "Frequency"], value="A-Z, a-z", id="dropdown_order_labels", clearable=False, style={"fontSize": "12px"}), width=12),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div("Image Order", style={"fontSize": "12px","marginBottom": "2px"}), width=12),
                    dbc.Col(dcc.Dropdown(dropdown_image_vals, value=["Similarity"], id="dropdown_order_images", clearable=False, multi=True, style={"fontSize": "12px"}), width=12),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Checklist(options=[{"label": " Marked first", "value": " Marked images first"}], value=[], id="check_marked_first", inline=True, style={"fontSize": "12px"}), width=6),
                    dbc.Col(dbc.Checklist(options=[{"label": " Hide relabeled", "value": " Hide relabeled images"}], value=[], id="check_discard_relabeled", inline=True, style={"fontSize": "12px"}), width=6),
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
            html.Div("Filtering", style={"fontWeight": "600","marginBottom": "10px","fontSize": "14px","color": "#333"}),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Confidence Range", style={"fontSize": "12px","marginBottom": "5px","textAlign": "center"}),
                            dcc.RangeSlider(0, 1, 0.05, value=[0, 1], id="probs_range_slider", marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                        ],
                        width=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        [
                            html.Div("Variance Range", style={"fontSize": "12px","marginBottom": "5px","textAlign": "center"}),
                            dcc.RangeSlider(0, 1, 0.05, value=[0, 1], id="vars_range_slider", marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                        ],
                        width=6,
                        id="col_vars_slider",
                        style={"display": "block" if HAS_VAR else "none"},
                        className="mb-3",
                    ),
                ]
            ),
            dbc.Row([dbc.Col(dbc.Button("Invert Selection", id="button_invert_marks", n_clicks=0, color="secondary", size="sm", style={"width": "100%"}), width=12)]),
        ]
    ),
    className="mb-3",
)

# --- Cluster UI (com update de artifacts) — FIX (1) + (5)
cluster_card = dbc.Card(
    dbc.CardBody(
        [
            html.Div("Clustering", style={"fontWeight": "600","marginBottom": "10px","fontSize": "14px","color": "#333"}),
            dbc.Row([dbc.Col(html.Div("Target Class", style={"fontSize": "12px","marginBottom": "2px"}), width=12)]),
            dbc.Row(
                [
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
            dbc.Row([dbc.Col(html.Div("Distance Cut", style={"fontSize": "12px","marginBottom": "5px"}), width=12)]),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Slider(
                            min=0.0, max=1.0, step=0.01, value=0.50,
                            id="slider_cluster_cut",
                            marks={0: {'label': '0', 'style': {'font-size': '12px'}},
                                   0.5: {'label': '0.5', 'style': {'font-size': '12px'}},
                                   1.0: {'label': '1', 'style': {'font-size': '12px'}}},
                            tooltip={"placement": "bottom", "always_visible": True, "template": "Cut: {value:.2f}"},
                        ),
                        width=12,
                        className="mb-3",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Button("Update Cluster", id="button_update_cluster", n_clicks=0, color="primary", size="sm", style={"width": "100%"}), width=6),
                    dbc.Col(dbc.Button("Apply Cluster Labels", id="button_relabel_cluster", n_clicks=0, color="success", size="sm", style={"width": "100%"}), width=6),
                ],
                className="mb-3",
            ),

            # >>> novo botão (artifacts) — FIX (5)
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Update cluster files (dirty classes)",
                            id="button_update_cluster_files",
                            n_clicks=0,
                            color="dark",
                            size="sm",
                            style={"width": "100%"},
                        ),
                        width=12,
                    ),
                ],
                className="mb-2",
            ),

            html.Div(
                id="cluster_summary",
                style={"marginTop": "8px","fontFamily": "monospace","whiteSpace": "pre-wrap","fontSize": "11px",
                       "padding": "8px","backgroundColor": "#f8f9fa","borderRadius": "3px","border": "1px solid #dee2e6"},
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
            html.Div("Save & Export", style={"fontWeight": "600","marginBottom": "10px","fontSize": "14px","color": "#333"}),
            dbc.Row([dbc.Col(html.Div("CSV Name", style={"fontSize": "12px","marginBottom": "2px"}), width=12)]),
            dbc.Row([dbc.Col(dbc.Input(value=csv_basename, id="input_save_csv", type="text", style={"fontSize": "12px"}), width=12, className="mb-2")]),
            dbc.Row(
                [
                    dbc.Col(dbc.Button("Save CSV", id="button_save_csv", n_clicks=0, color="primary", size="sm", style={"width": "100%"}), width=6),
                    dbc.Col(dbc.Button("UNDO", id="button_undo", n_clicks=0, color="warning", size="sm", style={"width": "100%"}), width=6),
                ],
                className="mb-3",
            ),
            dbc.Row([dbc.Col(html.Div("Dataset Name", style={"fontSize": "12px","marginBottom": "2px"}), width=12)]),
            dbc.Row([dbc.Col(dbc.Input(value="annotated_dataset", id="input_finish_work", type="text", style={"fontSize": "12px"}), width=12, className="mb-2")]),
            dbc.Row([dbc.Col(dbc.Button("Export Dataset", id="button_finish_work", n_clicks=0, color="success", size="sm", style={"width": "100%"}), width=12)]),
        ]
    ),
    className="mb-3",
)

IMAGES = create_list_dics(_list_src=[], _list_thumbnail=[], _list_name_figure=[])

fig_init = _get_overview_fig_cached(project_name, csv_folder, version=0, graph_range=graph_range)

# =========================
# LAYOUT
# =========================
app.layout = html.Div(
    [
        dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [
                        dbc.Col(html.Div(header, style={"textAlign": "left"}), width=8),
                        dbc.Col(
                            html.Div(
                                [html.Div("Select Batch", style={"fontSize": "12px","color": "#666","marginBottom": "4px"}), batch_dropdown],
                                style={"paddingTop": "15px"},
                            ),
                            width=4,
                        ),
                    ],
                    align="center",
                    className="mb-4 mt-3",
                ),

                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="g_scatter_plot",
                                                figure=fig_init,
                                                style={"height": "74vh", "minHeight": "500px"},
                                                config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage", "resetScale2d"], "displayModeBar": True},
                                            ),
                                        ],
                                        style={"padding": "10px"},
                                    ),
                                    style={"height": "100%", "border": "1px solid #dee2e6"},
                                ),
                            ],
                            width=7,
                            className="mb-4",
                        ),

                        dbc.Col(
                            [
                                html.Div(
                                    [allb_card, labeling_card, display_card, filtering_card, cluster_card, save_card],
                                    style={"height": "74vh", "minHeight": "500px", "overflowY": "auto", "paddingRight": "10px"},
                                ),
                            ],
                            width=5,
                            className="mb-4",
                        ),
                    ],
                    className="mb-4",
                ),

                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div("Image Gallery", style={"fontWeight": "600","marginBottom": "10px","fontSize": "14px","color": "#333"}),
                                            html.Div(
                                                imageselector.ImageSelector(
                                                    id="g_image_selector",
                                                    images=IMAGES,
                                                    galleryHeaderStyle={"position": "sticky","top": 0,"height": "0px","background": "#000000","zIndex": -1},
                                                ),
                                                style={"height": "44vh","minHeight": "300px","overflow": "auto","backgroundColor": background_color,"padding": "10px",
                                                       "borderRadius": "5px","border": "1px solid #dee2e6"},
                                            ),
                                        ]
                                    ),
                                    style={"border": "1px solid #dee2e6"},
                                ),
                            ],
                            width=12,
                        ),
                    ],
                    className="mb-4",
                ),

                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Alert(id="alert_allb", children="", is_open=False, color="info", dismissable=True, duration=6000),
                                dbc.Alert(id="alert_cluster_problem", children="", is_open=False, color="warning", dismissable=True, duration=8000),
                                # >>> novo alert — FIX (5)
                                dbc.Alert(id="alert_cluster_files", children="", is_open=False, color="dark", dismissable=True, duration=12000),
                            ],
                            width=12,
                            className="mb-3",
                        ),
                    ]
                ),
            ],
        ),

        # Hidden Stores
        dcc.Store(id="selected_custom_points", data=json.dumps([])),
        dcc.Store(id="unchecked_points", data=json.dumps([])),  # persistência cross-page (vamos atualizar no callback principal)
        dcc.Store(id="overview_hidden_groups", data=json.dumps([])),
        dcc.Store(id="state_store_df", data=df.to_json()),
        dcc.Store(id="store_context", data=store_context_init),
        dcc.Store(id="cluster_meta_store", data=json.dumps({"ready": False})),
        dcc.Store(id="allb_store", data=json.dumps({"active": False, "token": None, "page": 0, "page_size": 2000, "total": 0})),
        dcc.Store(id="dummy_csv_save", data=0),
        dcc.Store(id="dummy_finish_work", data=0),
        html.Div(id="output", style={"display": "none"}),
    ],
    style={"backgroundColor": "#f8f9fa", "minHeight": "100vh", "padding": "20px"},
)


# =========================
# OVERVIEW LEGEND SYNC (mantido)
# =========================
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
# HELPERS (mantidos)
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

# app.py  (PARTE 4/5) — CALLBACKS PRINCIPAIS
#  FIX (1) cluster via .npz + dendrogram truncado
#  FIX (2) persistência do “uncheck” cross-page
#  FIX (3) Apply label / Apply cluster labels em TODA a seleção (todas as páginas) no ALLB
#  FIX (5) botão “Update cluster files (dirty classes)”

# =========================
# CALLBACK: EXPORT DATASET (mantido do seu)
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

        project_csv_folder = s_ctx.get("csv_folder")
        if project_csv_folder is None:
            project_csv_folder = globals().get("csv_folder", "")

        out_root = str(s_input_finish_work_value)
        if not exists(out_root):
            mkdir(out_root)

        if is_overview:
            print(f"[INFO] Modo OVERVIEW ativo. Exportando TODOS os batches...")
            print(f"[INFO] CSV Folder: {project_csv_folder}")

            if not project_csv_folder or not exists(project_csv_folder):
                print(f"[ERRO] Pasta de CSVs não encontrada: {project_csv_folder}")
                return [0]

            csv_files = [f for f in _list_csv_candidates(project_csv_folder) if f != OVERVIEW_VALUE]
            if not csv_files:
                print("[ERRO] Nenhum arquivo CSV encontrado na pasta!")
                return [0]

            total_images = 0
            total_classes = set()

            for csv_bn in csv_files:
                try:
                    csv_path = join(project_csv_folder, csv_bn)
                    if not exists(csv_path):
                        print(f"[WARN] CSV não encontrado: {csv_path}")
                        continue

                    if csv_bn in BATCH_DF_CACHE:
                        df_batch = _cache_get_batch(csv_bn)
                        if isinstance(df_batch, pd.DataFrame):
                            df_batch = df_batch.copy()
                        else:
                            df_batch = pd.DataFrame()
                        print(f"  - Batch {csv_bn}: usando cache ({len(df_batch)} registros)")
                    else:
                        df_batch = _ensure_required_columns(read_input_csv(csv_path))
                        print(f"  - Batch {csv_bn}: lendo do disco ({len(df_batch)} registros)")

                    img_dir = _images_dir_for_csv(s_ctx, csv_bn)
                    if not exists(img_dir):
                        print(f"[WARN] Diretório de imagens não encontrado: {img_dir}")
                        continue

                    batch_images = 0
                    for _, row in df_batch.iterrows():
                        try:
                            label = str(row["manual_label"])
                            name = str(row["names"])

                            dst_dir = join(out_root, label)
                            if not exists(dst_dir):
                                mkdir(dst_dir)
                                total_classes.add(label)

                            src = join(img_dir, name)
                            dst = join(dst_dir, name)

                            if not exists(src):
                                continue
                            if not exists(dst):
                                shutil.copy2(src, dst)
                                total_images += 1
                                batch_images += 1
                        except Exception:
                            continue

                    print(f"    ✓ {batch_images} imagens exportadas")
                except Exception as e:
                    print(f"[ERRO] Erro ao processar batch {csv_bn}: {e}")
                    continue

            print(f"\n[SUCESSO] Dataset exportado (OVERVIEW). imgs={total_images} classes={len(total_classes)} out={out_root}\n")
        else:
            df_updated = pd.read_json(StringIO(s_store_df))

            if "csv_basename" in df_updated.columns:
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
                        if exists(src):
                            shutil.copy2(src, dst)
                    except Exception:
                        continue
            else:
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
                    except Exception:
                        pass

            print("\nDataset recorded.\n")

    return [0]


# =========================
# CALLBACK: SAVE CSV (mantido, + limpa DIRTY_BATCHES; DIRTY_CLASSES fica p/ update artifacts)
# =========================
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

        project_csv_folder = s_ctx.get("csv_folder") or globals().get("csv_folder", "")

        if is_overview and not allb_active:
            print("[INFO] OVERVIEW: salvando batches dirty...")
            if not DIRTY_BATCHES:
                print("[INFO] Nenhum batch modificado.")
                return [0]

            saved_count = 0
            for bn in sorted(DIRTY_BATCHES):
                try:
                    dfb = _cache_get_batch(bn)
                    if dfb is None:
                        continue
                    save_csv(dfb, bn)
                    saved_count += 1
                except Exception as e:
                    print(f"[WARN] save failed {bn}: {e}")

            DIRTY_BATCHES.clear()
            print(f"[OK] batches salvos={saved_count}")
            return [0]

        if allb_active:
            for bn in sorted(DIRTY_BATCHES):
                try:
                    dfb = _cache_get_batch(bn)
                    if dfb is None:
                        continue
                    save_csv(dfb, bn)
                except Exception as e:
                    print(f"[WARN] save dirty batch failed: {bn}: {e}")
            DIRTY_BATCHES.clear()
            return [0]

        df_updated = pd.read_json(StringIO(s_store_df))
        if "custom_data" in df_updated.columns:
            df_updated["custom_data"] = df_updated["custom_data"].astype(str)
        if "names" in df_updated.columns:
            df_updated["names"] = df_updated["names"].astype(str)

        csv_bn = str(s_ctx.get("csv_basename", ""))
        if csv_bn and csv_bn != OVERVIEW_VALUE:
            _cache_set_batch(csv_bn, df_updated)
            DIRTY_BATCHES.add(csv_bn)

        save_csv(df_updated, str(s_input_save_csv_value))
        DIRTY_BATCHES.discard(csv_bn)

    return [0]


# =========================
# CALLBACK: UPDATE CLUSTER FILES (DIRTY CLASSES) — FIX (5)
# =========================
@app.callback(
    Output("alert_cluster_files", "children"),
    Output("alert_cluster_files", "is_open"),
    Input("button_update_cluster_files", "n_clicks"),
    State("store_context", "data"),
    prevent_initial_call=True,
)
def update_cluster_files(n, s_ctx):
    if not n:
        return "", False

    s_ctx = s_ctx or {}
    proj = str(s_ctx.get("project_name", project_name))
    csv_folder_path = str(s_ctx.get("csv_folder", csv_folder))

    targets = set(DIRTY_CLASSES)
    if not targets:
        return "[cluster-files] Nenhuma classe marcada como dirty (nada a fazer).", True

    # conta tamanhos atuais (varredura leve)
    cur_counts = _count_classes_for_targets(csv_folder_path, targets)

    to_update: list[str] = []
    same: list[tuple[str, int]] = []
    missing: list[tuple[str, int]] = []

    deleted: list[tuple[str, int, int | None, str]] = []   # (cls, cur, art_n, path_or_reason)
    skipped_small: list[tuple[str, int, int | None]] = []  # (cls, cur, art_n)

    fail: list[tuple[str, str]] = []
    ok: list[str] = []

    # 1) Decide ação por classe dirty
    for cls in sorted(targets, key=lambda s: s.lower()):
        cur = int(cur_counts.get(cls, 0))
        art_n = _artifact_count(proj, cls)

        # >>> FIX: se classe não existe mais (cur==0) OU ficou pequena (cur<2),
        #          o artefato não deve ficar no diretório
        if cur < 2:
            del_ok, del_msg = _delete_cluster_artifact(proj, cls)
            if del_ok:
                # se havia algo ou já estava ausente, consideramos "limpo"
                deleted.append((cls, cur, art_n, del_msg))
                # como sua definição de "dirty" é baseada em mudança de quantidade,
                # e agora está <2, tiramos do dirty.
                DIRTY_CLASSES.discard(cls)
            else:
                fail.append((cls, f"[delete] {del_msg}"))
                # mantém dirty pra tentar de novo depois
            continue

        # cur >= 2 -> precisa existir/estar consistente
        if art_n is None:
            missing.append((cls, cur))
            to_update.append(cls)
        elif int(art_n) != cur:
            to_update.append(cls)
        else:
            same.append((cls, cur))
            # pela sua regra (diferença de quantidade), esta classe não precisa mais ficar dirty
            DIRTY_CLASSES.discard(cls)

    # 2) Gera/regera .npz para as que precisam
    for cls in to_update:
        success, out = _run_class_clustering(proj, one_class=cls)
        if success:
            ok.append(cls)
            DIRTY_CLASSES.discard(cls)
        else:
            fail.append((cls, (out[-1500:] if isinstance(out, str) else str(out))))
            # mantém dirty

    # 3) Mensagem
    msg_lines = []
    msg_lines.append(f"[cluster-files] project='{proj}' targets={len(targets)}")

    if same:
        msg_lines.append(f"  - já ok (sem mudança de N): {len(same)}")
    if missing:
        msg_lines.append(f"  - sem .npz (cur>=2): {len(missing)}")
    if ok:
        msg_lines.append(f"  - atualizadas/regeneradas: {len(ok)}")
    if deleted:
        # deleted inclui cur==0, cur==1 e também "missing" (já estava ausente)
        # vamos reportar separando “removidos de fato” de “já não existia”
        removed_real = [d for d in deleted if d[3] not in ("missing",)]
        already_missing = [d for d in deleted if d[3] == "missing"]
        if removed_real:
            msg_lines.append(f"  - artifacts removidos (classe sumiu/ficou <2): {len(removed_real)}")
        if already_missing:
            msg_lines.append(f"  - artifacts já ausentes (classe <2): {len(already_missing)}")

    if fail:
        msg_lines.append(f"  - falharam: {len(fail)}")

    # detalhe de algumas deleções/erros (curto)
    if deleted:
        show = deleted[:10]
        msg_lines.append("\n[deleted/cleaned] (até 10)")
        for cls, cur, art_n, why in show:
            msg_lines.append(f" - {cls} | cur={cur} | art_n={art_n} | {why}")

    if fail:
        msg_lines.append("\n[falhas] (até 6, últimas linhas)")
        for cls, logtail in fail[:6]:
            msg_lines.append(f"\n--- {cls} ---\n{logtail}")

    return "\n".join(msg_lines), True


# =========================
# CALLBACK: REFRESH opções do cluster dropdown
# =========================
@app.callback(
    Output("dropdown_cluster_class", "options"),
    Input("store_context", "data"),
    Input("state_store_df", "data"),
)
def refresh_class_options(s_ctx, s_store_df):
    try:
        _df = pd.read_json(StringIO(s_store_df))
        labs = sorted(_df["manual_label"].astype(str).unique().tolist(), key=lambda s: s.lower()) if "manual_label" in _df.columns else []
    except Exception:
        labs = []
    return [{"label": x, "value": x} for x in labs]


# =========================
# CALLBACK: UPDATE CLUSTER (carrega .npz, fcluster, dendro truncado) — FIX (1)
# =========================
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
    State("store_context", "data"),
    State("allb_store", "data"),
    State("unchecked_points", "data"),
)
def update_cluster_view(
    nclicks,
    class_label,
    cut01,
    s_store_df,
    s_selected_custom_points,
    s_ctx,
    s_allb_store_json,
    s_unchecked_points_json,
):
    if not nclicks:
        return json.dumps({"ready": False}), go.Figure(), "", "", False

    s_ctx = s_ctx or {}
    proj = str(s_ctx.get("project_name", project_name))
    allb_active = bool(s_ctx.get("allb_active", False))

    if not class_label:
        return json.dumps({"ready": False}), go.Figure(), "", "Selecione uma classe primeiro.", True

    # cut01 do slider (0..1)
    cut01f = float(cut01 if cut01 is not None else 0.5)
    cut01f = min(max(cut01f, 0.0), 1.0)

    # unchecked set (global ids)
    try:
        unchecked_list = json.loads(s_unchecked_points_json) if s_unchecked_points_json else []
        unchecked_set = set(str(x) for x in unchecked_list if x is not None)
    except Exception:
        unchecked_set = set()

    # token?
    try:
        allb_store = json.loads(s_allb_store_json) if s_allb_store_json else {}
    except Exception:
        allb_store = {}
    token = str(allb_store.get("token")) if (allb_active and allb_store.get("active") and allb_store.get("token")) else None

    # se classe está dirty, tenta regenerar antes
    if str(class_label) in DIRTY_CLASSES:
        _run_class_clustering(proj, one_class=str(class_label))

    # -------- ALLB --------
    if allb_active and token:
        df_sel = _allb_collect_selected_for_class(token, s_ctx, str(class_label), unchecked_set=unchecked_set)
        if df_sel.empty:
            return json.dumps({"ready": False}), go.Figure(), "", "Nenhum ponto dessa classe na seleção (ALLB).", True

        name2cid, meta_art = _cluster_ids_best(proj, str(class_label), cut01f, df_sel=df_sel)
        if "error" in meta_art:
            return json.dumps({"ready": False}), go.Figure(), "", meta_art["error"], True

        cids, gids = [], []
        miss = 0
        for nm, gid in zip(df_sel["names"].astype(str).tolist(), df_sel["custom_data"].astype(str).tolist()):
            cid = name2cid.get(str(nm))
            if cid is None:
                miss += 1
                continue
            cids.append(int(cid))
            gids.append(str(gid))

        if not gids:
            return json.dumps({"ready": False}), go.Figure(), "", "Artefato não contém nomes compatíveis com a seleção (ALLB).", True

        vc = pd.Series(cids).value_counts()
        K = int(vc.shape[0])
        max_sz = int(vc.max()) if not vc.empty else 0

        Z_for_fig = meta_art.get("Z", None)
        if Z_for_fig is None:
            art = _load_cluster_artifact(proj, str(class_label))
            Z_for_fig = art["Z"] if art else None
        fig = _build_dendrogram_figure(Z_for_fig, cut_dist=float(meta_art["cut_dist"]))

        summary = (
            f"[cluster] class='{class_label}' | selection(allb)={len(df_sel)} | used={len(gids)} | missing={miss}\n"
            f"[cluster] groups_in_selection={K} | max_imgs_in_group={max_sz}\n"
            f"[cluster-source] {meta_art.get('source','artifact')}\n"
            f"[artifact] N_total={meta_art.get('N_total')} | groups_total={meta_art.get('groups_total')} | mode={meta_art.get('mode')}\n"
            f"[artifact] metric={meta_art.get('metric')} | method={meta_art.get('method')} | pca_dim={meta_art.get('pca_dim')} | cut_mode={meta_art.get('cut_mode')}\n"
            f"cut={cut01f:.2f} (dist={meta_art.get('cut_dist'):.6g}/{meta_art.get('dist_max'):.6g})"
        )

        meta = {
            "ready": True,
            "source": str(meta_art.get("source", "artifact")),
            "project": proj,
            "class": str(class_label),
            "cut01": float(cut01f),
            "dist_max": float(meta_art.get("dist_max", 1.0)),
            "cut_dist": float(meta_art.get("cut_dist", 0.0)),
            "cut_mode": str(meta_art.get("cut_mode", "")),
            "groups_in_selection": int(K),
            "max_group_size": int(max_sz),
            "map": {g: int(c) for g, c in zip(gids, cids)},  # gid->cid
            "scope": "allb",
            "token": token,
        }
        return json.dumps(meta), fig, summary, "", False

    # -------- BATCH --------
    try:
        _df = pd.read_json(StringIO(s_store_df))
    except Exception as e:
        return json.dumps({"ready": False}), go.Figure(), "", f"Falha ao ler DF: {e}", True

    for c in ["custom_data", "names", "manual_label"]:
        if c in _df.columns:
            _df[c] = _df[c].astype(str)

    try:
        selected_points = json.loads(s_selected_custom_points) if s_selected_custom_points else []
        selected_points = [str(x) for x in selected_points if x is not None]
    except Exception:
        selected_points = []

    if not selected_points:
        return json.dumps({"ready": False}), go.Figure(), "", "Selecione pontos no scatter primeiro.", True

    df_sel = _df[_df["custom_data"].astype(str).isin(selected_points)].copy()
    df_sel = df_sel[df_sel["manual_label"].astype(str) == str(class_label)].copy()
    if df_sel.empty:
        return json.dumps({"ready": False}), go.Figure(), "", "Nenhum ponto dessa classe na seleção (batch).", True

    name2cid, meta_art = _cluster_ids_best(proj, str(class_label), cut01f, df_sel=df_sel)
    if "error" in meta_art:
        return json.dumps({"ready": False}), go.Figure(), "", meta_art["error"], True

    gids, cids = [], []
    miss = 0
    for nm, cd in zip(df_sel["names"].astype(str).tolist(), df_sel["custom_data"].astype(str).tolist()):
        cid = name2cid.get(str(nm))
        if cid is None:
            miss += 1
            continue
        gids.append(str(cd))
        cids.append(int(cid))

    if not gids:
        return json.dumps({"ready": False}), go.Figure(), "", "Artefato não contém nomes compatíveis com a seleção (batch).", True

    vc = pd.Series(cids).value_counts()
    K = int(vc.shape[0])
    max_sz = int(vc.max()) if not vc.empty else 0

    Z_for_fig = meta_art.get("Z", None)
    if Z_for_fig is None:
        art = _load_cluster_artifact(proj, str(class_label))
        Z_for_fig = art["Z"] if art else None
    fig = _build_dendrogram_figure(Z_for_fig, cut_dist=float(meta_art["cut_dist"]))

    summary = (
        f"[cluster] class='{class_label}' | selection(batch)={len(df_sel)} | used={len(gids)} | missing={miss}\n"
        f"[cluster] groups_in_selection={K} | max_imgs_in_group={max_sz}\n"
        f"[cluster-source] {meta_art.get('source','artifact')}\n"
        f"[artifact] N_total={meta_art.get('N_total')} | groups_total={meta_art.get('groups_total')} | mode={meta_art.get('mode')}\n"
        f"[artifact] metric={meta_art.get('metric')} | method={meta_art.get('method')} | pca_dim={meta_art.get('pca_dim')} | cut_mode={meta_art.get('cut_mode')}\n"
        f"cut={cut01f:.2f} (dist={meta_art.get('cut_dist'):.6g}/{meta_art.get('dist_max'):.6g})"
    )

    meta = {
        "ready": True,
        "source": str(meta_art.get("source", "artifact")),
        "project": proj,
        "class": str(class_label),
        "cut01": float(cut01f),
        "dist_max": float(meta_art.get("dist_max", 1.0)),
        "cut_dist": float(meta_art.get("cut_dist", 0.0)),
        "cut_mode": str(meta_art.get("cut_mode", "")),
        "groups_in_selection": int(K),
        "max_group_size": int(max_sz),
        "map": {g: int(c) for g, c in zip(gids, cids)},  # custom_data->cid
        "scope": "batch",
        "token": None,
    }
    return json.dumps(meta), fig, summary, "", False


# =========================
# CALLBACK: PRINCIPAL (mudança de batch / allb / apply label / apply cluster / pager) — FIX (2) + (3)
# =========================
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
    s_gallery_images,
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
):
    global labels_colors, next_label_id, csv_folder, UNDO_SNAPSHOT

    out_ctx = dict(s_ctx) if isinstance(s_ctx, dict) else {}
    out_ctx.setdefault("overview_version", 0)
    out_ctx.setdefault("is_overview", False)
    out_ctx.setdefault("allb_active", False)

    # allb_store
    try:
        allb_store = json.loads(s_allb_store_json) if s_allb_store_json else {"active": False}
    except Exception:
        allb_store = {"active": False}

    # DF atual (página allb ou batch)
    try:
        df_updated = pd.read_json(StringIO(s_store_df))
    except Exception:
        df_updated = pd.DataFrame()

    for c in ["custom_data", "names", "manual_label", "correct_label", "csv_basename", "custom_data_orig"]:
        if c in df_updated.columns:
            df_updated[c] = df_updated[c].astype(str)

    # unchecked_set (persistente)  — FIX (2)
    try:
        unchecked_list = json.loads(s_unchecked_points) if s_unchecked_points else []
        unchecked_set = set(str(x) for x in unchecked_list if x is not None)
    except Exception:
        unchecked_set = set()

    # helper: sincroniza unchecked_set a partir do estado atual da galeria (página atual)
    def _sync_unchecked_from_gallery():
        nonlocal unchecked_set
        if not s_gallery_images:
            return
        for it in s_gallery_images:
            gid = it.get("custom_data")
            if gid is None:
                continue
            gid = str(gid)
            sel = bool(it.get("isSelected", True))
            if sel:
                unchecked_set.discard(gid)
            else:
                unchecked_set.add(gid)

    ctx = dash.callback_context
    flag_callback = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

    df_store_updated = df_updated.to_json()
    selectedpoints = s_selected_custom_points if s_selected_custom_points else json.dumps([])
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
          - df_store_updated para página atual
        NÃO zera unchecked_set.  — FIX (2)
        """
        nonlocal df_updated, df_store_updated, allb_store, selectedpoints, out_alert, out_alert_open

        token = _current_allb_token()
        if not token:
            return

        _allb_recompute_batches(token, str(out_ctx.get("csv_folder", "")), changed_batches)
        total = _allb_total(token)

        page_size = int(allb_store.get("page_size", 2000))
        page = int(allb_store.get("page", 0))
        pages = (total + page_size - 1) // page_size if page_size > 0 else 1
        if page >= pages:
            page = max(0, pages - 1)

        allb_store["total"] = int(total)
        allb_store["page"] = int(page)

        df_page = _allb_make_page_df(token, page, page_size, out_ctx) if total > 0 else pd.DataFrame()
        df_updated = df_page.copy()
        df_store_updated = df_updated.to_json()
        selectedpoints = json.dumps([])

        out_alert = f"[allb] selection atualizado: total={total} | {_format_allb_page_info(allb_store)}"
        out_alert_open = True

    # =====================================================================
    # Troca batch via dropdown
    # =====================================================================
    if flag_callback == "dropdown_batch":
        chosen = str(i_dropdown_batch_value)

        # sempre sincroniza unchecked atual antes de trocar
        _sync_unchecked_from_gallery()

        if chosen == OVERVIEW_VALUE:
            out_ctx["is_overview"] = True
            out_ctx["allb_active"] = bool(allb_store.get("active") and allb_store.get("token"))
            out_header = "Image Labeling Tool - OVERVIEW"
            selectedpoints = json.dumps([])

            if out_ctx["allb_active"]:
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
                out_alert = "[OK] Overview ativado. Use lasso/box + 'Build selection'."
                out_alert_open = True

            return [
                selectedpoints,
                df_store_updated,
                json.dumps(sorted(unchecked_set)),
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

        # batch normal
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
                json.dumps(sorted(unchecked_set)),
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

        out_header = "Image Labeling Tool - " + stem
        out_input_save_csv = chosen_csv
        out_dropdown_opts = new_dropdown_opts
        out_dropdown_val = ["Similarity"]
        out_vars_style = {"display": "block"} if new_has_var else {"display": "none"}

        return [
            selectedpoints,
            df_store_updated,
            json.dumps(sorted(unchecked_set)),
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

    # =====================================================================
    # Seleção no scatter (batch normal)
    # =====================================================================
    if flag_callback == "g_scatter_plot":
        if bool(out_ctx.get("is_overview", False)):
            return [
                selectedpoints,
                df_store_updated,
                json.dumps(sorted(unchecked_set)),
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

    # =====================================================================
    # Build all-batches selection (a partir do overview)
    # =====================================================================
    elif flag_callback == "button_allb_build":
        # reset unchecked para nova seleção
        unchecked_set = set()

        out_ctx["is_overview"] = True
        out_ctx["allb_active"] = True

        geom = _parse_selection_polygon(i_selection_g_scatter_plot)
        if not geom:
            out_alert = "[ERRO] Faça uma seleção (lasso/box) no Overview antes de clicar em Build selection."
            out_alert_open = True
            return [
                selectedpoints,
                df_store_updated,
                json.dumps([]),
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

        visible_classes = _get_visible_classes_from_figure(s_g_scatter_plot_figure)

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
            visible_classes=visible_classes,
        )

        allb_store = {"active": True, "token": token, "page": 0, "page_size": page_size, "total": int(total)}
        out_ctx["allb_active"] = True

        if total <= 0:
            df_store_updated = pd.DataFrame().to_json()
            out_alert = "[OK] All-batches selection construída, mas não há pontos para essa seleção/filtros."
            out_alert_open = True
        else:
            df_page = _allb_make_page_df(token, 0, page_size, out_ctx)
            df_store_updated = df_page.to_json()
            out_alert = f"[OK] All-batches selection construída. total={total} | {_format_allb_page_info(allb_store)}"
            out_alert_open = True

        selectedpoints = json.dumps([])
        out_header = "Image Labeling Tool - OVERVIEW (allb active)"

    # =====================================================================
    # Paginação all-batches — FIX (2): persiste check/uncheck ao trocar
    # =====================================================================
    elif flag_callback in ("button_allb_prev", "button_allb_next"):
        if not allb_store.get("active") or not allb_store.get("token"):
            out_alert = "[ERRO] All-batches selection não está ativo. Use Build selection primeiro."
            out_alert_open = True
        else:
            # 1) captura estado atual da página e atualiza unchecked_set
            _sync_unchecked_from_gallery()

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
            out_alert = f"[OK] {_format_allb_page_info(allb_store)}"
            out_alert_open = True
            out_header = "Image Labeling Tool - OVERVIEW (allb active)"

    # =====================================================================
    # Apply label — FIX (3): no ALLB aplica em TODA a seleção (token - unchecked)
    # =====================================================================
    elif flag_callback == "button_aplicar_novo_label":
        new_label = str(s_input_aplicar_novo_label_value) if s_input_aplicar_novo_label_value is not None else ""
        if not new_label.strip():
            out_msg = "[ERRO] Label vazio."
        else:
            # sempre sincroniza o estado atual da página
            _sync_unchecked_from_gallery()

            if bool(out_ctx.get("allb_active", False)) and allb_store.get("active") and allb_store.get("token"):
                token = str(allb_store.get("token"))
                cache = ALLB_CACHE.get(token)
                if not cache:
                    out_msg = "[ERRO] token allb inválido/expirado."
                else:
                    batch_order = cache.get("batch_order", [])
                    packmasks = cache.get("packmasks", {})
                    lens = cache.get("lens", {})
                    csv_folder_local = str(out_ctx.get("csv_folder", ""))

                    unchecked_by_bn = _unchecked_by_batch(unchecked_set)

                    changed_batches = set()
                    undo_batches = {}
                    col_id = int(_ensure_color_for_label(new_label))

                    # aplica em TODOS os selecionados pelo token (menos unchecked)
                    for bn in batch_order:
                        pack = packmasks.get(bn, None)
                        n = int(lens.get(bn, 0))
                        if pack is None or n <= 0:
                            continue

                        mask_bits = np.unpackbits(np.frombuffer(pack, dtype=np.uint8), count=n).astype(bool)
                        if not mask_bits.any():
                            continue

                        csv_path = join(csv_folder_local, bn)
                        if not exists(csv_path):
                            continue

                        dfb = _cache_get_batch(bn)
                        if dfb is None:
                            dfb = _ensure_required_columns(read_input_csv(csv_path))

                        # ajusta tamanho caso CSV mudou
                        if len(dfb) < len(mask_bits):
                            mask_bits = mask_bits[:len(dfb)]
                        elif len(dfb) > len(mask_bits):
                            mask2 = np.zeros(len(dfb), dtype=bool)
                            mask2[:len(mask_bits)] = mask_bits
                            mask_bits = mask2

                        # remove unchecked desse batch (por custom_data_orig)
                        bad_cd = unchecked_by_bn.get(bn, set())
                        if bad_cd:
                            badmask = dfb["custom_data"].astype(str).isin(list(bad_cd)).to_numpy(dtype=bool)
                            mask_bits = mask_bits & (~badmask)

                        if not mask_bits.any():
                            continue

                        # undo snapshot (por batch)
                        if bn not in undo_batches:
                            undo_batches[bn] = dfb.copy()

                        old_labs = dfb.loc[mask_bits, "manual_label"].astype(str).unique().tolist()
                        dfb.loc[mask_bits, "manual_label"] = new_label
                        dfb.loc[mask_bits, "colors"] = col_id

                        _cache_set_batch(bn, dfb)
                        DIRTY_BATCHES.add(bn)
                        changed_batches.add(bn)

                        # marca dirty classes (p/ cluster files)
                        _mark_dirty_from_changes(old_labs, [new_label])

                    UNDO_SNAPSHOT = {"mode": "allb", "batches": undo_batches, "page_df": df_updated.copy()}

                    if s_check_save_csv_value == [" Save CSV after labeling"]:
                        for bn in sorted(changed_batches):
                            try:
                                dfb = _cache_get_batch(bn)
                                if dfb is not None:
                                    save_csv(dfb, bn)
                            except Exception:
                                pass

                    _bump_overview_version()
                    if changed_batches:
                        _refresh_allb_after_batch_changes(sorted(changed_batches))

                    out_header = "Image Labeling Tool - OVERVIEW (allb active)"
                    out_alert = f"[OK] Apply label (ALLB): batches={len(changed_batches)} label='{new_label}'"
                    out_alert_open = True

            else:
                # batch normal (igual ao seu)
                try:
                    selected_and_checked = []
                    selected_not_checked = []
                    if s_gallery_images:
                        for point in s_gallery_images:
                            cd = point.get("custom_data")
                            if cd is None:
                                continue
                            if point.get("isSelected") is True:
                                selected_and_checked.append(str(cd))
                            else:
                                selected_not_checked.append(str(cd))
                except Exception:
                    selected_and_checked, selected_not_checked = [], []

                UNDO_SNAPSHOT = {"mode": "batch", "csv_basename": str(out_ctx.get("csv_basename", "")), "df_before": df_updated.copy()}

                col_id = int(_ensure_color_for_label(new_label))
                old_labs = df_updated.loc[df_updated["custom_data"].isin(selected_and_checked), "manual_label"].astype(str).unique().tolist()

                df_updated.loc[df_updated["custom_data"].isin(selected_and_checked), "manual_label"] = new_label
                df_updated.loc[df_updated["custom_data"].isin(selected_and_checked), "colors"] = col_id

                csv_bn = str(out_ctx.get("csv_basename", ""))
                if csv_bn and csv_bn != OVERVIEW_VALUE:
                    _cache_set_batch(csv_bn, df_updated)
                    DIRTY_BATCHES.add(csv_bn)

                _mark_dirty_from_changes(old_labs, [new_label])
                _bump_overview_version()

                if s_check_save_csv_value == [" Save CSV after labeling"]:
                    save_csv(df_updated, str(s_input_save_csv_value))

                df_store_updated = df_updated.to_json()

                if s_check_discard_marked_value == [" Discard images after labeling"]:
                    selectedpoints = json.dumps(selected_not_checked)

    # =====================================================================
    # Apply cluster labels — FIX (3): no ALLB aplica em TODA a seleção (mapeamento do cluster_meta)
    # =====================================================================
    elif flag_callback == "button_relabel_cluster":
        # sincroniza unchecked atual da página
        _sync_unchecked_from_gallery()

        try:
            meta = json.loads(s_cluster_meta_json) if s_cluster_meta_json else {}
        except Exception:
            meta = {}

        if not meta or not meta.get("ready"):
            out_msg = "[ERRO] Cluster meta não está pronto. Clique em 'Update Cluster' primeiro."
        else:
            class_label = str(meta.get("class", ""))
            mapping = meta.get("map", {}) if isinstance(meta.get("map", {}), dict) else {}
            scope = str(meta.get("scope", ""))

            if not class_label or not mapping:
                out_msg = "[ERRO] Meta de cluster inválida."
            else:
                if bool(out_ctx.get("allb_active", False)) and scope == "allb":
                    token = _current_allb_token()
                    changed_batches = set()
                    undo_batches = {}
                    undo_page = df_updated.copy()

                    unchecked_set_local = set(unchecked_set)

                    # mapping é gid->cid; aplica em TODOS os gids do mapping, exceto unchecked
                    by_batch: dict[str, dict[int, list[str]]] = {}
                    for gid, cid in mapping.items():
                        gid = str(gid)
                        if gid in unchecked_set_local:
                            continue
                        sp = _split_global_id(gid)
                        if not sp:
                            continue
                        bn, cd_orig = sp
                        by_batch.setdefault(bn, {}).setdefault(int(cid), []).append(cd_orig)

                    csv_folder_local = str(out_ctx.get("csv_folder", ""))
                    for bn, groups in by_batch.items():
                        csv_path = join(csv_folder_local, bn)
                        if not exists(csv_path):
                            continue

                        dfb = _cache_get_batch(bn)
                        if dfb is None:
                            dfb = _ensure_required_columns(read_input_csv(csv_path))

                        undo_batches[bn] = dfb.copy()

                        # old labels afetados (tipicamente a classe base)
                        # pega apenas linhas que vão mudar
                        to_change_mask = np.zeros(len(dfb), dtype=bool)
                        for cid, cds in groups.items():
                            to_change_mask |= dfb["custom_data"].astype(str).isin([str(x) for x in cds]).to_numpy(dtype=bool)

                        old_labs = dfb.loc[to_change_mask, "manual_label"].astype(str).unique().tolist()

                        # aplica por grupo (define label novo)
                        for cid, cds in groups.items():
                            new_lab = f"{class_label}__g{int(cid)}"
                            col_id = int(_ensure_color_for_label(new_lab))
                            m = dfb["custom_data"].astype(str).isin([str(x) for x in cds])
                            if m.any():
                                dfb.loc[m, "manual_label"] = new_lab
                                dfb.loc[m, "colors"] = col_id
                                _mark_dirty_class(new_lab)

                        _mark_dirty_from_changes(old_labs, [class_label])
                        _cache_set_batch(bn, dfb)
                        DIRTY_BATCHES.add(bn)
                        changed_batches.add(bn)

                    UNDO_SNAPSHOT = {"mode": "allb", "batches": undo_batches, "page_df": undo_page}

                    if s_check_save_csv_value == [" Save CSV after labeling"]:
                        for bn in sorted(changed_batches):
                            try:
                                dfb = _cache_get_batch(bn)
                                if dfb is not None:
                                    save_csv(dfb, bn)
                            except Exception:
                                pass

                    _bump_overview_version()
                    if token and changed_batches:
                        _refresh_allb_after_batch_changes(sorted(changed_batches))

                    out_alert = f"[OK] Apply Cluster Labels (ALLB): batches={len(changed_batches)}."
                    out_alert_open = True

                else:
                    # batch normal: aplica só nos itens selecionados (como antes)
                    try:
                        selected_and_checked = []
                        selected_not_checked = []
                        if s_gallery_images:
                            for point in s_gallery_images:
                                cd = point.get("custom_data")
                                if cd is None:
                                    continue
                                if point.get("isSelected") is True:
                                    selected_and_checked.append(str(cd))
                                else:
                                    selected_not_checked.append(str(cd))
                    except Exception:
                        selected_and_checked, selected_not_checked = [], []

                    UNDO_SNAPSHOT = {"mode": "batch", "csv_basename": str(out_ctx.get("csv_basename", "")), "df_before": df_updated.copy()}

                    changed = 0
                    missing = 0
                    old_labs = []
                    if "manual_label" in df_updated.columns:
                        old_labs = df_updated.loc[df_updated["custom_data"].isin(selected_and_checked), "manual_label"].astype(str).unique().tolist()

                    for cd in selected_and_checked:
                        cid = mapping.get(cd, None)
                        if cid is None:
                            missing += 1
                            continue
                        new_lab = f"{class_label}__g{int(cid)}"
                        df_updated.loc[df_updated["custom_data"] == cd, "manual_label"] = new_lab
                        df_updated.loc[df_updated["custom_data"] == cd, "colors"] = int(_ensure_color_for_label(new_lab))
                        _mark_dirty_class(new_lab)
                        changed += 1

                    csv_bn = str(out_ctx.get("csv_basename", ""))
                    if csv_bn and csv_bn != OVERVIEW_VALUE:
                        _cache_set_batch(csv_bn, df_updated)
                        DIRTY_BATCHES.add(csv_bn)

                    _mark_dirty_from_changes(old_labs, [class_label])
                    _bump_overview_version()

                    df_store_updated = df_updated.to_json()
                    if s_check_save_csv_value == [" Save CSV after labeling"]:
                        save_csv(df_updated, str(s_input_save_csv_value))
                    if s_check_discard_marked_value == [" Discard images after labeling"]:
                        selectedpoints = json.dumps(selected_not_checked)

                    out_msg = f"[OK] Relabel by cluster: alteradas={changed} | sem_cluster_id={missing}."

    # =====================================================================
    # UNDO (mantido)
    # =====================================================================
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
                        _cache_set_batch(bn, df_updated)
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
                            _cache_set_batch(str(bn), dfb)
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

    # =====================================================================
    # Invert marks — agora persiste globalmente (unchecked_set)
    # =====================================================================
    elif flag_callback == "button_invert_marks":
    # 1) primeiro, captura qualquer alteração manual na página atual
        _sync_unchecked_from_gallery()

    # 2) Se ALLB estiver ativo: inverte a marcação de TODA a seleção (todas as páginas)
        if bool(out_ctx.get("allb_active", False)) and allb_store.get("active") and allb_store.get("token"):
            token = str(allb_store.get("token"))
            cache = ALLB_CACHE.get(token)

            if cache:
                csv_folder_local = str(out_ctx.get("csv_folder", ""))
                batch_order = cache.get("batch_order", [])
                packmasks = cache.get("packmasks", {})
                lens = cache.get("lens", {})

            # percorre toda a seleção do token e alterna (checked <-> unchecked)
                for bn in batch_order:
                    pack = packmasks.get(bn, None)
                    n = int(lens.get(bn, 0))
                    if pack is None or n <= 0:
                        continue

                    try:
                        mask_bits = np.unpackbits(np.frombuffer(pack, dtype=np.uint8), count=n).astype(bool)
                    except Exception:
                        continue

                    if not mask_bits.any():
                        continue

                    idx_sel = np.flatnonzero(mask_bits)
                    if idx_sel.size == 0:
                        continue

                    csv_path = join(csv_folder_local, bn)
                    if not exists(csv_path):
                        continue

                # leitura mínima para mapear idx -> custom_data
                    try:
                        dfb = _read_batch_minimal(csv_path, bn)
                    except Exception:
                        continue
                    if dfb is None or dfb.empty:
                        continue

                    idx_sel = idx_sel[idx_sel < len(dfb)]
                    if idx_sel.size == 0:
                        continue

                    cds = dfb.iloc[idx_sel]["custom_data"].astype(str).tolist()
                    for cd_orig in cds:
                        gid = _global_id(bn, cd_orig)
                        if gid in unchecked_set:
                            unchecked_set.discard(gid)  # estava desmarcado -> marcar
                        else:
                            unchecked_set.add(gid)      # estava marcado -> desmarcar

            out_alert = "[OK] Invert Selection aplicado em TODA a seleção (todas as páginas)."
            out_alert_open = True

    # 3) Caso NÃO seja ALLB (batch normal): inverte apenas os itens exibidos na galeria atual
        else:
            if s_gallery_images:
                for it in s_gallery_images:
                    gid = it.get("custom_data")
                    if gid is None:
                        continue
                    gid = str(gid)
                    if gid in unchecked_set:
                        unchecked_set.discard(gid)
                    else:
                        unchecked_set.add(gid)

            out_alert = "[OK] Invert Selection aplicado na galeria atual."
            out_alert_open = True

    # --- outputs comuns ---
    unchecked_out = json.dumps(sorted(unchecked_set))
    allb_store_out = json.dumps(allb_store)
    allb_info = _format_allb_page_info(allb_store)

    return [
        selectedpoints,
        df_store_updated,
        unchecked_out,
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

# app.py  (PARTE 5/5) — scatter_plot_image_selector + RUN
#  FIX (2) persistência de “uncheck” cross-page (regra: UI atual prevalece; senão usa unchecked_set)
#  (resto do arquivo vem das PARTES 1–4)

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

    # DF atual (page DF do allb ou batch normal)
    try:
        _df = pd.read_json(StringIO(i_store_df))
    except Exception:
        _df = pd.DataFrame()

    for c in ["custom_data", "names", "manual_label", "correct_label", "csv_basename", "custom_data_orig"]:
        if c in _df.columns:
            _df[c] = _df[c].astype(str)

    # unchecked_set persistente
    try:
        unchecked_points = json.loads(i_unchecked_points) if i_unchecked_points else []
        unchecked_points = [str(x) for x in unchecked_points if x is not None]
    except Exception:
        unchecked_points = []
    unchecked_set = set(unchecked_points)

    # estado atual da UI (página atual) — isso permite “re-check” mesmo se ainda estiver no unchecked_set
    gallery_map = {}
    if s_g_image_selector_images:
        for it in s_g_image_selector_images:
            cd = it.get("custom_data")
            if cd is not None:
                gallery_map[str(cd)] = bool(it.get("isSelected", True))

    # =========================
    # OVERVIEW MODE
    # =========================
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

        # ordem multi-critério dentro da página
        order_vals = i_dropdown_order_images_value
        if isinstance(order_vals, str):
            order_vals = [order_vals]
        if not order_vals:
            order_vals = ["A-Z, a-z"]

        filtered_df = _df.copy()

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

        # ===== FIX (2): selected flag
        # regra: se está na UI atual, respeita UI; senão, usa unchecked_set
        sel_flags = []
        cds = filtered_df["custom_data"].astype(str).tolist()
        for cd in cds:
            if cd in gallery_map:
                sel_flags.append(bool(gallery_map[cd]))
            else:
                sel_flags.append(cd not in unchecked_set)

        filtered_df["selected"] = sel_flags

        if i_check_marked_first_value == [" Marked images first"]:
            checked_df = filtered_df[filtered_df["selected"] == True]
            unchecked_df = filtered_df[filtered_df["selected"] == False]
            filtered_df = pd.concat([checked_df, unchecked_df], ignore_index=True)

        # paths por linha (por batch)
        srcs, thumbs = [], []
        for _, row in filtered_df.iterrows():
            bn = str(row.get("csv_basename", ""))
            name = str(row.get("names", ""))
            img_dir = _images_dir_for_csv(s_ctx, bn) if bn else str(s_ctx.get("path_to_images", ""))
            srcs.append(_abs_to_assets_url(join(img_dir, name)))
            thumbs.append(_abs_to_assets_url(join(img_dir, name)))

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

    # =========================
    # NORMAL BATCH MODE
    # =========================
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

    # ===== FIX (2): selected flag (batch normal)
    # regra: UI atual prevalece; senão, unchecked_set decide
    sel_flags = []
    for cd in ordered_df["custom_data"].astype(str).tolist():
        if cd in gallery_map:
            sel_flags.append(bool(gallery_map[cd]))
        else:
            sel_flags.append(cd not in unchecked_set)
    ordered_df["selected"] = sel_flags

    if i_check_marked_first_value == [" Marked images first"]:
        checked_df = ordered_df[ordered_df["selected"] == True]
        unchecked_df = ordered_df[ordered_df["selected"] == False]
        ordered_df = pd.concat([checked_df, unchecked_df], ignore_index=True)

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

