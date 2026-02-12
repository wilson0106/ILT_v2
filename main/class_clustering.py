#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
class_clustering.py

Gera artefatos de clustering por classe em:
  main/assets/<project>/clusters/<class>.npz

Compatível com o data_explorer.py:
  - names (N,)  -> nomes reais das imagens
  - linkage (N-1,4)
  - dist_max (float)

Fonte dos embeddings:
  - CSVs em main/assets/<project>/dataframes/*.csv
  - colunas: names, embed_file, embed_idx, e uma coluna de classe:
      preferencialmente manual_label; se for boolean/ruim, tenta preds1, depois pred.
  - embeddings em main/assets/<project>/embeddings/*.npz
      chaves comuns: feats / embeddings / vectors, e (opcional) names

Para classes grandes: usa centroides (MiniBatchKMeans) e "expande" o linkage para N folhas
sem custo O(N^2) de distâncias completas.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# --- scipy: linkage/fcluster ---
try:
    from scipy.cluster.hierarchy import linkage
except Exception as e:
    linkage = None

# --- sklearn: PCA + MiniBatchKMeans ---
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
except Exception:
    PCA = None
    MiniBatchKMeans = None


# ======================================================================================
# Utils
# ======================================================================================

def _now() -> float:
    return time.perf_counter()


def _log(msg: str):
    print(msg, flush=True)


def _sanitize_filename(x: str) -> str:
    x = str(x)
    x = x.replace("/", "_").replace("\\", "_").strip()
    x = re.sub(r"\s+", "_", x)
    return x


def _looks_booleanish(s: pd.Series) -> bool:
    # True/False em bool, ou strings "True"/"False", ou 0/1 com baixa diversidade
    if s.dtype == bool:
        return True
    vals = s.dropna().astype(str).str.lower().unique()
    if len(vals) == 0:
        return True
    if set(vals).issubset({"true", "false", "0", "1"}):
        # se só tem 1 valor e é "true", costuma ser bug de pipeline
        return True
    return False


def _pick_label_column(df: pd.DataFrame) -> str:
    # tenta manual_label, depois preds1, depois pred
    candidates = ["manual_label", "preds1", "pred"]
    for c in candidates:
        if c in df.columns:
            if c == "manual_label" and _looks_booleanish(df[c]):
                continue
            return c
    # fallback: a primeira coluna "categoria" razoável
    for c in df.columns:
        if c in ("names", "x", "y", "embed_file", "embed_idx"):
            continue
        if df[c].dtype == object:
            return c
    raise RuntimeError("Não encontrei coluna de classe (manual_label/preds1/pred).")


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def _resolve_embed_path(project_dir: Path, embed_file_value: str) -> Path:
    """
    embed_file no CSV pode vir:
      - só o nome (batch_0001.npz)
      - relativo (embeddings/batch_0001.npz)
      - absoluto
    """
    p = Path(str(embed_file_value))
    if p.is_absolute() and p.exists():
        return p
    # relativo ao projeto
    cand1 = project_dir / p
    if cand1.exists():
        return cand1
    # relativo à pasta embeddings
    cand2 = project_dir / "embeddings" / p.name
    if cand2.exists():
        return cand2
    # tenta também caso embed_file já venha com "embeddings/..."
    if "embeddings" in p.parts:
        tail = Path(*p.parts[p.parts.index("embeddings")+1:])
        cand3 = project_dir / "embeddings" / tail
        if cand3.exists():
            return cand3
    raise FileNotFoundError(f"embed_file não encontrado: '{embed_file_value}' (projeto={project_dir})")


def _load_feats_from_npz(npz_path: Path) -> np.ndarray:
    """
    Suporta npz com chaves:
      - feats (preferido)
      - embeddings
      - vectors
      - X
    """
    z = np.load(npz_path, allow_pickle=False)
    keys = list(z.keys())

    for k in ("feats", "embeddings", "vectors", "X"):
        if k in keys:
            arr = z[k]
            break
    else:
        # pega o primeiro array 2D
        arr = None
        for k in keys:
            a = z[k]
            if isinstance(a, np.ndarray) and a.ndim == 2:
                arr = a
                break
        if arr is None:
            raise KeyError(f"{npz_path.name}: não achei array 2D de embeddings (keys={keys})")

    if arr.ndim != 2:
        raise ValueError(f"{npz_path.name}: embeddings não são 2D (shape={arr.shape})")

    return arr


# ======================================================================================
# Core: build X per class
# ======================================================================================

def _read_index_table(dataframes_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """
    Lê todos CSVs e retorna um DF com colunas mínimas:
      names, embed_file, embed_idx, label_col
    """
    csvs = sorted([p for p in dataframes_dir.glob("*.csv") if p.is_file()])
    if not csvs:
        raise FileNotFoundError(f"Sem CSVs em {dataframes_dir}")

    # descobre colunas a partir do primeiro CSV
    df0 = pd.read_csv(csvs[0], nrows=50)
    label_col = _pick_label_column(df0)

    need_cols = ["names", "embed_file", "embed_idx", label_col]
    missing = [c for c in need_cols if c not in df0.columns]
    if missing:
        raise RuntimeError(f"CSV não tem colunas necessárias {missing}. Colunas disponíveis: {list(df0.columns)}")

    parts = []
    for p in csvs:
        t0 = _now()
        df = pd.read_csv(
            p,
            usecols=need_cols,
            dtype={"names": "string", "embed_file": "string"},
            low_memory=False,
        )
        # embed_idx pode vir float/string -> força int
        df["embed_idx"] = pd.to_numeric(df["embed_idx"], errors="coerce").fillna(-1).astype(np.int64)

        # label sempre string (evita bool/NaN)
        df[label_col] = df[label_col].astype("string")

        # remove linhas inválidas
        df = df[(df["embed_idx"] >= 0) & df["names"].notna() & df["embed_file"].notna() & df[label_col].notna()]
        parts.append(df)

        if verbose:
            _log(f"[index] {p.name}: {len(df)} linhas (t={_now()-t0:.2f}s)")

    out = pd.concat(parts, ignore_index=True)
    out.attrs["label_col"] = label_col
    return out


def _gather_class_matrix(
    df_index: pd.DataFrame,
    project_dir: Path,
    class_label: str,
    embed_cache: Dict[str, np.ndarray],
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna:
      X: (N,D) float64
      names: (N,) object/string
    """
    label_col = df_index.attrs["label_col"]
    dfc = df_index[df_index[label_col] == str(class_label)]
    if dfc.empty:
        return np.empty((0, 0), dtype=np.float64), np.array([], dtype=object)

    # ordem final
    names = dfc["names"].astype(str).to_numpy(dtype=object)
    embed_files = dfc["embed_file"].astype(str).to_numpy(dtype=object)
    embed_idxs = dfc["embed_idx"].to_numpy(dtype=np.int64)

    # agrupa por embed_file para carregar e indexar em bloco
    # map file -> list of positions
    file_to_pos: Dict[str, List[int]] = {}
    for pos, ef in enumerate(embed_files):
        file_to_pos.setdefault(ef, []).append(pos)

    # descobre D pelo primeiro arquivo
    first_file = next(iter(file_to_pos.keys()))
    first_path = _resolve_embed_path(project_dir, first_file)
    first_key = str(first_path)
    if first_key not in embed_cache:
        embed_cache[first_key] = _load_feats_from_npz(first_path)
    D = int(embed_cache[first_key].shape[1])

    N = len(names)
    X = np.empty((N, D), dtype=np.float64)

    bad = 0
    for ef, positions in file_to_pos.items():
        ep = _resolve_embed_path(project_dir, ef)
        k = str(ep)
        if k not in embed_cache:
            t0 = _now()
            embed_cache[k] = _load_feats_from_npz(ep)
            if verbose:
                _log(f"[embed] load {ep.name} shape={embed_cache[k].shape} (t={_now()-t0:.2f}s)")

        feats = embed_cache[k]
        # índices para esse arquivo
        idxs = embed_idxs[np.array(positions, dtype=np.int64)]

        # valida faixa
        ok_mask = (idxs >= 0) & (idxs < feats.shape[0])
        if not np.all(ok_mask):
            bad += int((~ok_mask).sum())
            positions_ok = np.array(positions, dtype=np.int64)[ok_mask]
            idxs_ok = idxs[ok_mask]
        else:
            positions_ok = np.array(positions, dtype=np.int64)
            idxs_ok = idxs

        if len(positions_ok) == 0:
            continue

        X[positions_ok, :] = feats[idxs_ok, :].astype(np.float64, copy=False)

    if bad > 0:
        _log(f"[WARN] classe='{class_label}': {bad} linhas com embed_idx fora do range foram ignoradas (ficaram lixo em X).")
        # remove essas linhas pra evitar NaNs/lixo
        # (refaz com máscara segura)
        mask_ok = np.ones(N, dtype=bool)
        # revalida todos de forma segura (custo O(N))
        for i in range(N):
            try:
                ep = _resolve_embed_path(project_dir, embed_files[i])
                feats = embed_cache[str(ep)]
                if not (0 <= int(embed_idxs[i]) < feats.shape[0]):
                    mask_ok[i] = False
            except Exception:
                mask_ok[i] = False
        X = X[mask_ok]
        names = names[mask_ok]

    return X, names


# ======================================================================================
# Clustering: full vs centroid-expanded linkage
# ======================================================================================

def _apply_pca_if_needed(X: np.ndarray, pca_dim: int, verbose: bool = False) -> np.ndarray:
    if pca_dim <= 0:
        return X
    if PCA is None:
        if verbose:
            _log("[PCA] sklearn não disponível -> pulando PCA.")
        return X

    N, D = X.shape
    ncomp = int(min(pca_dim, N, D))
    if ncomp < 2 or ncomp >= D:
        return X

    t0 = _now()
    pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=0)
    Xr = pca.fit_transform(X)
    if verbose:
        _log(f"[PCA] ajustado: pedido={pca_dim} -> usando={ncomp} (N={N}, D={D}) (t={_now()-t0:.2f}s)")
    return Xr.astype(np.float64, copy=False)


def _centroid_expanded_linkage(
    X: np.ndarray,
    method: str,
    metric: str,
    max_centroids: int,
    verbose: bool = False
) -> np.ndarray:
    """
    Faz clustering hierárquico nos centroides e expande para N folhas:
      - merges internos (dentro de cada centroid) a dist=0
      - merges entre centroides usando linkage(centroids)

    Retorna Z_full (N-1,4) válido para scipy/fcluster/dendrogram.
    """
    if linkage is None:
        raise RuntimeError("scipy não disponível (linkage).")
    if MiniBatchKMeans is None:
        raise RuntimeError("sklearn não disponível (MiniBatchKMeans).")

    N, D = X.shape
    if N < 2:
        return np.zeros((0, 4), dtype=np.float64)

    # regra simples para K (capado em max_centroids)
    K = int(min(max_centroids, max(2, int(np.sqrt(N) * 20))))
    K = min(K, N)
    if verbose:
        _log(f"[centroid] N={N} -> K={K} centroides")

    t0 = _now()
    km = MiniBatchKMeans(
        n_clusters=K,
        random_state=0,
        batch_size=2048,
        max_iter=200,
        n_init=3,
        reassignment_ratio=0.01,
    )
    labels = km.fit_predict(X)
    C = km.cluster_centers_.astype(np.float64, copy=False)
    C = _l2_normalize(C)

    if verbose:
        _log(f"[centroid] kmeans done (t={_now()-t0:.2f}s)")

    # linkage nos centroides
    t1 = _now()
    Zc = linkage(C, method=method, metric=metric)
    if verbose:
        _log(f"[centroid] linkage(centroids) done: Zc={Zc.shape} (t={_now()-t1:.2f}s)")

    # --- expande para N folhas ---
    # Ordena por labels para varrer grupos contíguos
    order = np.argsort(labels, kind="mergesort")
    labels_sorted = labels[order]

    Z_full = np.zeros((N - 1, 4), dtype=np.float64)

    # tamanhos de clusters (índices até 2N-2)
    sizes = np.ones((2 * N,), dtype=np.int64)

    centroid_leaf = np.empty((K,), dtype=np.int64)

    next_cluster = N
    row = 0

    # varre grupos
    start = 0
    for k in range(K):
        # acha slice do centroid k
        while start < N and labels_sorted[start] < k:
            start += 1
        end = start
        while end < N and labels_sorted[end] == k:
            end += 1

        members = order[start:end]  # índices 0..N-1
        m = int(len(members))

        if m <= 0:
            # centroid vazio (raro em kmeans, mas pode ocorrer)
            centroid_leaf[k] = -1
            continue

        if m == 1:
            centroid_leaf[k] = int(members[0])
        else:
            cur = int(members[0])
            cur_size = 1
            for j in members[1:]:
                newc = int(next_cluster)
                Z_full[row, 0] = float(cur)
                Z_full[row, 1] = float(int(j))
                Z_full[row, 2] = 0.0
                Z_full[row, 3] = float(cur_size + 1)
                sizes[newc] = cur_size + 1

                cur = newc
                cur_size += 1
                next_cluster += 1
                row += 1
            centroid_leaf[k] = int(cur)

        start = end

    # remove centroides vazios (se existirem)
    valid = centroid_leaf >= 0
    if not np.all(valid):
        centroid_leaf = centroid_leaf[valid]
        C = C[valid]
        K = int(len(centroid_leaf))
        if K < 2:
            # sem merges entre centroides, mas ainda precisa de Z_full com N-1 rows
            # aqui, tudo já foi unido internamente e não há merges adicionais:
            # faltariam merges -> inviável. Melhor cair pro full linkage (vai doer) ou abortar.
            raise RuntimeError("Centroides válidos < 2; não dá pra construir linkage expandido.")

        # recomputa Zc com centroides válidos
        Zc = linkage(C, method=method, metric=metric)

    base = next_cluster  # primeiro cluster criado na parte centroid
    # agora row deveria ser N-K (merges internos)
    # completa com merges entre centroides
    for r in range(Zc.shape[0]):
        a = int(Zc[r, 0])
        b = int(Zc[r, 1])
        dist = float(Zc[r, 2])

        if a < K:
            ag = int(centroid_leaf[a])
        else:
            ag = int(base + (a - K))

        if b < K:
            bg = int(centroid_leaf[b])
        else:
            bg = int(base + (b - K))

        newg = int(N + (row))  # regra do scipy: cluster novo do row atual é N+row
        # mas como row já está contando rows em Z_full, isso coincide com base+r
        # (mantemos consistência usando newg explícito)
        size_new = int(sizes[ag] + sizes[bg])
        Z_full[row, 0] = float(ag)
        Z_full[row, 1] = float(bg)
        Z_full[row, 2] = dist
        Z_full[row, 3] = float(size_new)
        sizes[newg] = size_new
        row += 1

    if row != (N - 1):
        raise RuntimeError(f"Linkage expandido inconsistente: rows={row}, esperado={N-1}")

    return Z_full


def _full_linkage(X: np.ndarray, method: str, metric: str, verbose: bool = False) -> np.ndarray:
    if linkage is None:
        raise RuntimeError("scipy não disponível (linkage).")
    t0 = _now()
    Z = linkage(X, method=method, metric=metric)
    if verbose:
        _log(f"[full] linkage done: Z={Z.shape} (t={_now()-t0:.2f}s)")
    return Z


# ======================================================================================
# Main
# ======================================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project", help="nome do projeto em main/assets/<project>/")
    ap.add_argument("--class", dest="one_class", default=None, help="processa apenas esta classe")
    ap.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"], help="métrica do linkage")
    ap.add_argument("--linkage", dest="method", default="average",
                    choices=["average", "complete", "single", "ward"],
                    help="método do linkage (ward só faz sentido com euclidean)")
    ap.add_argument("--pca-dim", type=int, default=64, help="dimensão PCA (0 desliga). default=64")
    ap.add_argument("--max-full", type=int, default=1500, help="até N<=max_full usa linkage completo. default=1500")
    ap.add_argument("--max-centroids", type=int, default=1500, help="limite de centroides no fallback. default=1500")
    ap.add_argument("--min-samples", type=int, default=2, help="mínimo de amostras por classe. default=2")
    ap.add_argument("--verbose", action="store_true", help="logs detalhados")
    args = ap.parse_args()

    project = args.project
    method = args.method
    metric = args.metric

    if method == "ward" and metric != "euclidean":
        raise SystemExit("ward requer metric=euclidean (caso contrário, distâncias ficam inválidas).")

    cwd = Path(os.getcwd())
    project_dir = cwd / "main" / "assets" / project
    dataframes_dir = project_dir / "dataframes"
    clusters_dir = project_dir / "clusters"

    if not dataframes_dir.is_dir():
        raise SystemExit(f"[ERROR] dataframes não encontrado: {dataframes_dir}")

    clusters_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[clustering] project={project}")
    _log(f"[clustering] metric={metric} linkage={method} pca_dim={args.pca_dim} max_full={args.max_full} max_centroids={args.max_centroids}")

    t_all = _now()
    df_index = _read_index_table(dataframes_dir, verbose=args.verbose)
    label_col = df_index.attrs["label_col"]
    _log(f"[clustering] label_col='{label_col}' | rows={len(df_index)}")

    classes = sorted(df_index[label_col].dropna().astype(str).unique().tolist())
    if args.one_class is not None:
        classes = [args.one_class]
        _log(f"[clustering] --class={args.one_class}")

    else:
        _log(f"[clustering] --class ausente -> rodando para {len(classes)} classes.")

    embed_cache: Dict[str, np.ndarray] = {}  # cache de feats por arquivo

    ok = 0
    fail = 0

    for cls in classes:
        t0 = _now()
        _log(f"\n[clustering] class={cls}")

        try:
            X, names = _gather_class_matrix(df_index, project_dir, cls, embed_cache, verbose=args.verbose)
            N = int(X.shape[0])
            if N < args.min_samples:
                _log(f"[skip] N={N} < min_samples={args.min_samples}")
                continue

            _log(f"[clustering] selected rows: {N}")

            # normaliza (para cosine)
            X = _l2_normalize(X)

            # PCA (opcional)
            Xr = _apply_pca_if_needed(X, args.pca_dim, verbose=args.verbose)
            if Xr is not X:
                Xr = _l2_normalize(Xr)

            # linkage: full ou centroid fallback
            if N <= args.max_full:
                Z = _full_linkage(Xr, method=method, metric=metric, verbose=args.verbose)
                mode = "full"
            else:
                Z = _centroid_expanded_linkage(
                    Xr,
                    method=method,
                    metric=metric,
                    max_centroids=args.max_centroids,
                    verbose=args.verbose
                )
                mode = "centroid"

            dist_max = float(np.max(Z[:, 2])) if Z.size else 1.0

            out_path = clusters_dir / f"{_sanitize_filename(cls)}.npz"
            np.savez_compressed(
                out_path,
                names=names.astype(object),
                linkage=Z.astype(np.float64),
                dist_max=np.float64(dist_max),
                mode=np.array(mode),
                metric=np.array(metric),
                method=np.array(method),
                pca_dim=np.int32(args.pca_dim),
                max_full=np.int32(args.max_full),
            )

            _log(f"[OK] saved: {out_path} | mode={mode} | dist_max={dist_max:.6g} | t={_now()-t0:.2f}s")
            ok += 1

        except Exception as e:
            fail += 1
            _log(f"[ERROR] Falhou na classe '{cls}': {e} (t={_now()-t0:.2f}s)")
            continue

    _log(f"\n[done] ok={ok} fail={fail} total={len(classes)} | t_total={_now()-t_all:.2f}s")


if __name__ == "__main__":
    main()

