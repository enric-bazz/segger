"""Lightweight validation metrics for Segger outputs.

This module provides fast, reference-light metrics intended for quick model
selection and single-run quality checks.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import polars as pl
from scipy import sparse
from scipy.spatial import cKDTree

from ..io import StandardTranscriptFields
from .me_genes import load_me_genes_from_scrna


def valid_cell_id_expr(cell_id_column: str) -> pl.Expr:
    """Expression selecting rows with a valid cell identifier."""
    cell = pl.col(cell_id_column)
    cell_str = cell.cast(pl.Utf8)
    return (
        cell.is_not_null()
        & (cell_str != "-1")
        & (cell_str.str.to_uppercase() != "UNASSIGNED")
        & (cell_str.str.to_uppercase() != "NONE")
    )


def assigned_cell_expr(cell_id_column: str = "segger_cell_id") -> pl.Expr:
    """Expression selecting transcripts assigned to a valid cell."""
    return valid_cell_id_expr(cell_id_column)


_ENSEMBL_WITH_VERSION_RE = re.compile(r"^ENS[A-Z0-9]+(?:\.[0-9]+)$", re.IGNORECASE)
_REFERENCE_CELLTYPE_COLUMN_ALIASES = (
    "cell_type",
    "celltype",
    "cell_type_fine",
    "celltype_major",
    "annotation",
    "annot",
    "cell_label",
    "cluster_annotation",
)


def _normalize_gene_token(gene: object) -> str:
    """Normalize gene names for robust cross-reference matching.

    Notes
    -----
    - Case-insensitive matching avoids human/mouse symbol casing drift.
    - Version suffixes on Ensembl IDs (e.g. ENSG... .12) are stripped.
    """
    token = str(gene).strip()
    if not token:
        return ""
    if _ENSEMBL_WITH_VERSION_RE.match(token):
        token = token.rsplit(".", 1)[0]
    return token.upper()


def _build_gene_index_map(genes: Sequence[object]) -> tuple[dict[str, int], dict[str, int]]:
    """Build exact and normalized gene -> index maps."""
    exact: dict[str, int] = {}
    normalized: dict[str, int] = {}
    for idx, raw_gene in enumerate(genes):
        gene = str(raw_gene)
        # Keep first index for stable behavior across duplicates.
        if gene not in exact:
            exact[gene] = idx
        norm = _normalize_gene_token(gene)
        if norm and norm not in normalized:
            normalized[norm] = idx
    return exact, normalized


def _resolve_reference_celltype_column(
    ref: ad.AnnData,
    requested: str,
) -> str | None:
    """Resolve requested cell-type column with alias fallback."""
    if requested in ref.obs.columns:
        return requested
    for candidate in _REFERENCE_CELLTYPE_COLUMN_ALIASES:
        if candidate in ref.obs.columns:
            return candidate
    return None


def _looks_like_positional_var_names(genes: Sequence[str]) -> bool:
    """Heuristic for integer-like positional var_names (e.g. 0..N)."""
    if len(genes) == 0:
        return False
    sample = [str(g).strip() for g in genes[: min(len(genes), 256)]]
    numeric = sum(1 for token in sample if token.isdigit())
    return numeric >= max(1, int(0.9 * len(sample)))


def _reference_gene_names(
    ref: ad.AnnData,
    feature_column: str = "feature_name",
) -> list[str]:
    """Return robust reference gene names, preferring feature column when needed."""
    var_names = [str(g) for g in ref.var_names]
    if feature_column not in ref.var.columns:
        return var_names

    if not _looks_like_positional_var_names(var_names):
        return var_names

    feature_values = ref.var[feature_column].astype("string").fillna("")
    promoted: list[str] = []
    for fallback, raw in zip(var_names, feature_values.astype(str).tolist()):
        token = raw.strip()
        if not token or token.lower() in {"nan", "none"}:
            promoted.append(fallback)
        else:
            promoted.append(token)
    return promoted


def _effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size for non-negative weights."""
    w = np.asarray(weights, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return float("nan")
    sw = float(w.sum())
    sw2 = float(np.square(w).sum())
    if sw <= 0 or sw2 <= 0:
        return float("nan")
    return (sw * sw) / sw2


def _weighted_mean_ci95(values: np.ndarray, weights: np.ndarray) -> float:
    """Approximate 95% CI half-width for a weighted mean."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    v = v[mask]
    w = w[mask]
    if v.size == 0:
        return float("nan")
    mu = float(np.average(v, weights=w))
    neff = _effective_sample_size(w)
    if not np.isfinite(neff) or neff <= 1:
        return float("nan")
    var = float(np.average(np.square(v - mu), weights=w))
    se = np.sqrt(max(var, 0.0) / neff)
    return float(1.96 * se)


def _weighted_bernoulli_ci95(flags: np.ndarray, weights: np.ndarray) -> float:
    """Approximate 95% CI half-width for weighted Bernoulli proportion."""
    f = np.asarray(flags, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(f) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    f = f[mask]
    w = w[mask]
    if f.size == 0:
        return float("nan")
    p = float(np.average(f, weights=w))
    neff = _effective_sample_size(w)
    if not np.isfinite(neff) or neff <= 1:
        return float("nan")
    se = np.sqrt(max(p * (1.0 - p), 0.0) / neff)
    return float(1.96 * se)


def _binomial_pct_ci95(successes: int, total: int) -> float:
    """95% CI half-width in percentage points for a binomial proportion."""
    if total <= 0:
        return float("nan")
    p = min(max(float(successes) / float(total), 0.0), 1.0)
    se = np.sqrt(max(p * (1.0 - p), 0.0) / float(total))
    return float(100.0 * 1.96 * se)


def load_source_transcripts(source_path: Path) -> pl.DataFrame:
    """Load standardized source transcripts with only needed columns."""
    tx_fields = StandardTranscriptFields()
    source_path = Path(source_path)
    tx = None

    # Optional SpatialData input support
    try:
        from ..io.spatialdata_loader import is_spatialdata_path, load_from_spatialdata

        if is_spatialdata_path(source_path):
            tx_lf, _ = load_from_spatialdata(
                source_path,
                boundary_type="all",
                normalize=True,
            )
            tx = tx_lf.collect() if isinstance(tx_lf, pl.LazyFrame) else tx_lf
    except Exception:
        tx = None

    if tx is None:
        from ..io import get_preprocessor

        pp = get_preprocessor(source_path, min_qv=0, include_z=True)
        tx = pp.transcripts
        if isinstance(tx, pl.LazyFrame):
            tx = tx.collect()

    keep_cols = [
        tx_fields.row_index,
        tx_fields.feature,
        tx_fields.x,
        tx_fields.y,
    ]
    if tx_fields.z in tx.columns:
        keep_cols.append(tx_fields.z)
    if tx_fields.cell_id in tx.columns:
        keep_cols.append(tx_fields.cell_id)
    if tx_fields.compartment in tx.columns:
        keep_cols.append(tx_fields.compartment)
    tx = tx.select([c for c in keep_cols if c in tx.columns])

    if tx_fields.row_index not in tx.columns:
        tx = tx.with_row_index(name=tx_fields.row_index)

    return tx


def load_segmentation(segmentation_path: Path) -> pl.DataFrame:
    """Load segmentation parquet with canonical columns."""
    seg = pl.read_parquet(segmentation_path)
    required = ["row_index", "segger_cell_id"]
    missing = [c for c in required if c not in seg.columns]
    if missing:
        raise ValueError(
            f"Segmentation file missing required columns {missing}: {segmentation_path}"
        )
    return seg.select([c for c in ["row_index", "segger_cell_id", "segger_similarity"] if c in seg.columns])


def compute_assignment_metrics(
    seg_df: pl.DataFrame,
    cell_id_column: str = "segger_cell_id",
) -> dict[str, float]:
    """Compute transcript assignment coverage metrics and split fragment objects."""
    total = int(seg_df.height)
    if total == 0:
        return {
            "transcripts_total": 0,
            "transcripts_assigned": 0,
            "transcripts_assigned_pct": float("nan"),
            "transcripts_assigned_pct_ci95": float("nan"),
            "cells_assigned": 0,
            "fragments_assigned": 0,
        }

    assigned_df = seg_df.filter(assigned_cell_expr(cell_id_column))
    assigned = int(assigned_df.height)
    if assigned > 0:
        assigned_ids = assigned_df.select(pl.col(cell_id_column).cast(pl.Utf8).alias("_cell_id"))
        fragments = int(
            assigned_ids
            .filter(pl.col("_cell_id").str.starts_with("fragment-"))
            .select(pl.col("_cell_id").n_unique())
            .to_series()
            .item()
        )
        cells = int(
            assigned_ids
            .filter(~pl.col("_cell_id").str.starts_with("fragment-"))
            .select(pl.col("_cell_id").n_unique())
            .to_series()
            .item()
        )
    else:
        fragments = 0
        cells = 0
    return {
        "transcripts_total": total,
        "transcripts_assigned": assigned,
        "transcripts_assigned_pct": 100.0 * assigned / total,
        "transcripts_assigned_pct_ci95": _binomial_pct_ci95(assigned, total),
        "cells_assigned": cells,
        "fragments_assigned": fragments,
    }


def count_cells_from_anndata(anndata_path: Optional[Path]) -> Optional[int]:
    """Return number of cells (n_obs) from AnnData, or None if unavailable."""
    if anndata_path is None:
        return None
    path = Path(anndata_path)
    if not path.exists():
        return None

    adata = ad.read_h5ad(path, backed="r")
    try:
        return int(adata.n_obs)
    finally:
        try:
            if getattr(adata, "isbacked", False):
                adata.file.close()
        except Exception:
            pass


def merge_assigned_transcripts(
    seg_df: pl.DataFrame,
    source_tx: pl.DataFrame,
    cell_id_column: str = "segger_cell_id",
    row_index_column: str = "row_index",
) -> pl.DataFrame:
    """Inner-join source transcripts with assigned segmentation rows."""
    left = source_tx
    right = seg_df

    if row_index_column not in left.columns:
        left = left.with_row_index(name=row_index_column)

    left = left.with_columns(pl.col(row_index_column).cast(pl.Int64))
    right = right.with_columns(pl.col(row_index_column).cast(pl.Int64))
    right = right.filter(assigned_cell_expr(cell_id_column)).select([row_index_column, cell_id_column])

    return left.join(right, on=row_index_column, how="inner")


def _empty_resolvi_metrics() -> dict[str, float]:
    """Default empty return payload for RESOLVI-like contamination metric."""
    return {
        "resolvi_contamination_pct_fast": float("nan"),
        "resolvi_contamination_ci95_fast": float("nan"),
        "resolvi_contaminated_cells_pct_fast": float("nan"),
        "resolvi_contaminated_cells_pct_ci95_fast": float("nan"),
        "resolvi_metric_cells_used": 0,
        "resolvi_shared_genes_used": 0,
        "resolvi_cell_types_used": 0,
    }


def _build_cell_gene_matrix(
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str,
    feature_column: str,
    x_column: str,
    y_column: str,
    min_transcripts_per_cell: int,
    max_cells: int,
    seed: int,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, list[str]] | None:
    """Build sparse cell x gene counts with centroids and per-cell weights."""
    req = [cell_id_column, feature_column, x_column, y_column]
    for col in req:
        if col not in assigned_tx.columns:
            return None

    df = (
        assigned_tx.select(req)
        .drop_nulls()
        .with_columns(
            pl.col(cell_id_column).cast(pl.Utf8),
            pl.col(feature_column).cast(pl.Utf8),
        )
    )
    if df.height == 0:
        return None

    cell_stats = (
        df.group_by(cell_id_column)
        .agg(
            pl.len().alias("n_total"),
            pl.col(x_column).mean().alias("cx"),
            pl.col(y_column).mean().alias("cy"),
        )
        .filter(pl.col("n_total") >= int(min_transcripts_per_cell))
    )
    if cell_stats.height == 0:
        return None

    if max_cells > 0 and cell_stats.height > max_cells:
        rng = np.random.default_rng(seed)
        ids = np.asarray(cell_stats.get_column(cell_id_column).to_list(), dtype=object)
        picked = rng.choice(ids, size=max_cells, replace=False).tolist()
        cell_stats = cell_stats.filter(pl.col(cell_id_column).is_in(picked))

    cell_stats = cell_stats.sort(cell_id_column).with_row_index(name="_cid")
    if cell_stats.height == 0:
        return None

    df = df.join(cell_stats.select([cell_id_column, "_cid"]), on=cell_id_column, how="inner")
    if df.height == 0:
        return None

    gene_idx = (
        df.select(feature_column)
        .unique()
        .sort(feature_column)
        .with_row_index(name="_gid")
    )
    if gene_idx.height == 0:
        return None

    mapped = df.join(gene_idx, on=feature_column, how="inner")
    counts = mapped.group_by(["_cid", "_gid"]).agg(pl.len().alias("_count"))
    if counts.height == 0:
        return None

    rows = counts.get_column("_cid").to_numpy().astype(np.int64, copy=False)
    cols = counts.get_column("_gid").to_numpy().astype(np.int64, copy=False)
    data = counts.get_column("_count").to_numpy().astype(np.float64, copy=False)

    X = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(int(cell_stats.height), int(gene_idx.height)),
    ).tocsr()

    centroids = cell_stats.select(["cx", "cy"]).to_numpy().astype(np.float64, copy=False)
    weights = cell_stats.get_column("n_total").to_numpy().astype(np.float64, copy=False)
    gene_names = [str(g) for g in gene_idx.get_column(feature_column).to_list()]
    return X, centroids, weights, gene_names


def _load_reference_type_profiles(
    scrna_reference_path: Path,
    scrna_celltype_column: str,
    seg_gene_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load per-celltype expression profiles on genes shared with segmentation."""
    if not Path(scrna_reference_path).exists():
        return None

    ref = ad.read_h5ad(scrna_reference_path)
    resolved_celltype_column = _resolve_reference_celltype_column(ref, scrna_celltype_column)
    if resolved_celltype_column is None:
        return None

    labels_raw = ref.obs[resolved_celltype_column].astype(str).to_numpy()
    labels = np.asarray([str(x) for x in labels_raw], dtype=object)
    label_norm = np.char.lower(labels.astype(str))
    valid = (
        (labels.astype(str) != "")
        & (label_norm != "nan")
        & (label_norm != "none")
        & (label_norm != "-1")
    )
    if not np.any(valid):
        return None

    ref_genes = np.asarray(_reference_gene_names(ref, feature_column="feature_name"), dtype=object)
    ref_gene_to_idx_exact, ref_gene_to_idx_norm = _build_gene_index_map(ref_genes.tolist())

    seg_shared_idx: list[int] = []
    ref_shared_idx: list[int] = []
    for i, g in enumerate(seg_gene_names):
        sg = str(g)
        j = ref_gene_to_idx_exact.get(sg)
        if j is None:
            j = ref_gene_to_idx_norm.get(_normalize_gene_token(sg))
        if j is None:
            continue
        seg_shared_idx.append(i)
        ref_shared_idx.append(int(j))

    if len(seg_shared_idx) == 0:
        return None

    X_ref = ref.X
    if sparse.issparse(X_ref):
        X_ref = X_ref.tocsr()[valid][:, np.asarray(ref_shared_idx, dtype=np.int64)]
    else:
        X_ref = np.asarray(X_ref)[valid][:, np.asarray(ref_shared_idx, dtype=np.int64)]

    labels_valid = labels[valid].astype(str)
    type_names, type_inverse = np.unique(labels_valid, return_inverse=True)
    if type_names.size == 0:
        return None

    n_types = int(type_names.size)
    n_genes = int(len(seg_shared_idx))
    profiles = np.zeros((n_types, n_genes), dtype=np.float64)

    for t in range(n_types):
        idx = np.where(type_inverse == t)[0]
        if idx.size == 0:
            continue
        if sparse.issparse(X_ref):
            sub = X_ref[idx]
            profiles[t] = np.asarray(sub.mean(axis=0)).ravel()
        else:
            profiles[t] = np.asarray(X_ref[idx], dtype=np.float64).mean(axis=0)

    profiles = np.nan_to_num(profiles, nan=0.0, posinf=0.0, neginf=0.0)
    profiles = np.maximum(profiles, 0.0)
    keep = np.asarray(profiles.sum(axis=1)).ravel() > 0
    if not np.any(keep):
        return None

    return (
        np.asarray(seg_shared_idx, dtype=np.int64),
        profiles[keep],
        type_names[keep],
    )


def _prepare_reference_alignment_fast(
    assigned_tx: pl.DataFrame,
    *,
    scrna_reference_path: Optional[Path],
    scrna_celltype_column: str,
    cell_id_column: str,
    feature_column: str,
    x_column: str,
    y_column: str,
    min_transcripts_per_cell: int,
    max_cells: int,
    seed: int,
) -> dict[str, object] | None:
    """Build aligned per-cell matrix plus inferred host types from a reference."""
    if scrna_reference_path is None:
        return None

    built = _build_cell_gene_matrix(
        assigned_tx,
        cell_id_column=cell_id_column,
        feature_column=feature_column,
        x_column=x_column,
        y_column=y_column,
        min_transcripts_per_cell=min_transcripts_per_cell,
        max_cells=max_cells,
        seed=seed,
    )
    if built is None:
        return None

    X, centroids, cell_weights, gene_names = built
    if X.shape[0] == 0 or X.shape[1] == 0:
        return None

    ref_data = _load_reference_type_profiles(
        Path(scrna_reference_path),
        scrna_celltype_column=scrna_celltype_column,
        seg_gene_names=gene_names,
    )
    if ref_data is None:
        return None

    seg_shared_idx, ref_profiles, type_names = ref_data
    if seg_shared_idx.size == 0 or ref_profiles.size == 0:
        return None

    X = X[:, seg_shared_idx]
    shared_gene_names = [str(gene_names[int(i)]) for i in seg_shared_idx]
    if X.shape[1] == 0:
        return None

    totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float64, copy=False)
    keep = np.isfinite(totals) & (totals > 0)
    if not np.any(keep):
        return None
    if not np.all(keep):
        rows_keep = np.where(keep)[0]
        X = X[rows_keep]
        centroids = centroids[rows_keep]
        cell_weights = cell_weights[rows_keep]
        totals = totals[rows_keep]

    ref = np.asarray(ref_profiles, dtype=np.float64)
    ref_norm = np.linalg.norm(ref, axis=1)
    ref_norm[~np.isfinite(ref_norm) | (ref_norm <= 0)] = 1.0

    cell_norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    cell_norm[~np.isfinite(cell_norm) | (cell_norm <= 0)] = 1.0

    sim = X @ ref.T
    if sparse.issparse(sim):
        sim = sim.toarray()
    sim = np.asarray(sim, dtype=np.float64)
    sim /= cell_norm[:, None]
    sim /= ref_norm[None, :]
    host_type = np.argmax(sim, axis=1).astype(np.int64)

    return {
        "X": X,
        "centroids": centroids,
        "cell_weights": np.asarray(cell_weights, dtype=np.float64),
        "totals": totals,
        "gene_names": shared_gene_names,
        "ref_profiles": ref,
        "type_names": np.asarray(type_names, dtype=object),
        "host_type": host_type,
    }


def _discover_unique_markers_from_profiles(
    ref_profiles: np.ndarray,
    *,
    n_markers_per_type: int = 12,
    min_specificity_ratio: float = 1.5,
) -> list[np.ndarray]:
    """Pick type-specific markers from per-type reference profiles."""
    ref = np.asarray(ref_profiles, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[0] == 0 or ref.shape[1] == 0:
        return []

    n_types, n_genes = ref.shape
    best_type = np.argmax(ref, axis=0).astype(np.int64)
    best_val = ref[best_type, np.arange(n_genes, dtype=np.int64)]

    masked = ref.copy()
    masked[best_type, np.arange(n_genes, dtype=np.int64)] = 0.0
    second_val = masked.max(axis=0)

    ratio = best_val / np.maximum(second_val, 1e-9)
    score = best_val * np.log1p(np.maximum(ratio, 0.0))

    markers: list[np.ndarray] = []
    for t in range(n_types):
        idx = np.where(
            (best_type == t)
            & (best_val > 0)
            & np.isfinite(best_val)
            & np.isfinite(ratio)
        )[0]
        if idx.size == 0:
            markers.append(np.empty((0,), dtype=np.int64))
            continue

        specific = idx[ratio[idx] >= float(min_specificity_ratio)]
        chosen = specific if specific.size > 0 else idx
        order = np.argsort(score[chosen])[::-1]
        markers.append(chosen[order][: int(n_markers_per_type)].astype(np.int64, copy=False))

    return markers


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity clipped to [0, 1]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    sim = float(np.dot(a, b) / denom)
    if not np.isfinite(sim):
        return float("nan")
    return float(np.clip(sim, 0.0, 1.0))


def _simple_geometry_table(
    tx: pl.DataFrame,
    *,
    cell_id_column: str,
    x_column: str,
    y_column: str,
) -> pl.DataFrame:
    """Compute simple per-cell geometry from transcript coordinates."""
    req = [cell_id_column, x_column, y_column]
    if any(col not in tx.columns for col in req):
        return pl.DataFrame()

    df = (
        tx.select(req)
        .drop_nulls()
        .filter(valid_cell_id_expr(cell_id_column))
        .with_columns(pl.col(cell_id_column).cast(pl.Utf8))
    )
    if df.height == 0:
        return pl.DataFrame()

    geom = (
        df.group_by(cell_id_column)
        .agg(
            pl.len().alias("n_total"),
            pl.col(x_column).min().alias("x_min"),
            pl.col(x_column).max().alias("x_max"),
            pl.col(y_column).min().alias("y_min"),
            pl.col(y_column).max().alias("y_max"),
        )
        .with_columns(
            (pl.col("x_max") - pl.col("x_min")).alias("width"),
            (pl.col("y_max") - pl.col("y_min")).alias("height"),
        )
        .filter((pl.col("width") > 0) & (pl.col("height") > 0))
        .with_columns(
            (pl.col("width") * pl.col("height")).alias("area"),
            (
                pl.max_horizontal("width", "height")
                / (pl.min_horizontal("width", "height") + 1e-9)
            ).alias("elongation"),
            (2.0 * (pl.col("width") + pl.col("height"))).alias("perimeter_bbox"),
        )
        .with_columns(
            (
                4.0 * np.pi * pl.col("area")
                / ((pl.col("perimeter_bbox") * pl.col("perimeter_bbox")) + 1e-9)
            ).alias("circularity")
        )
        .select([cell_id_column, "n_total", "area", "elongation", "circularity"])
    )
    return geom


def _binary_jaccard_from_csr(
    X: sparse.csr_matrix,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary Jaccard similarity matrix for a sparse rows x genes matrix."""
    if X.shape[0] == 0 or X.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)

    B = X.copy().tocsr()
    if B.nnz == 0:
        return (
            np.zeros((X.shape[1], X.shape[1]), dtype=np.float64),
            np.zeros((X.shape[1],), dtype=np.float64),
        )
    B.data = np.ones_like(B.data)
    occur = np.asarray(B.sum(axis=0)).ravel().astype(np.float64, copy=False)
    co = (B.T @ B).astype(np.float64)
    if sparse.issparse(co):
        co = co.toarray()
    denom = occur[:, None] + occur[None, :] - co + float(eps)
    return co / denom, occur


def _spatial_gene_scores_fast(
    source_tx: pl.DataFrame,
    *,
    feature_column: str,
    x_column: str,
    y_column: str,
    radius: float,
    max_transcripts: int,
    min_gene_count: int,
    seed: int,
) -> dict[str, object] | None:
    """Approximate local spatial co-occurrence scores for gene pairs."""
    req = [feature_column, x_column, y_column]
    if any(col not in source_tx.columns for col in req):
        return None

    df = (
        source_tx.select(req)
        .drop_nulls()
        .with_columns(pl.col(feature_column).cast(pl.Utf8))
    )
    if df.height < 2:
        return None

    gene_counts = (
        df.group_by(feature_column)
        .agg(pl.len().alias("_n"))
        .filter(pl.col("_n") >= int(min_gene_count))
    )
    if gene_counts.height < 2:
        return None

    df = df.join(gene_counts.select([feature_column]), on=feature_column, how="inner")
    if df.height < 2:
        return None

    if max_transcripts > 0 and df.height > max_transcripts:
        df = df.sample(n=int(max_transcripts), seed=int(seed))

    gene_index = (
        df.select(feature_column)
        .unique()
        .sort(feature_column)
        .with_row_index(name="_gid")
    )
    if gene_index.height < 2:
        return None

    mapped = df.join(gene_index, on=feature_column, how="inner")
    codes = mapped.get_column("_gid").to_numpy().astype(np.int64, copy=False)
    coords = mapped.select([x_column, y_column]).to_numpy().astype(np.float64, copy=False)
    if coords.shape[0] < 2:
        return None

    tree = cKDTree(coords)
    neigh = tree.query_ball_point(coords, r=float(radius))

    rows: list[int] = []
    cols: list[int] = []
    for i, nbrs in enumerate(neigh):
        if not nbrs:
            continue
        nbr_codes = np.unique(codes[np.asarray(nbrs, dtype=np.int64)])
        rows.extend([i] * int(nbr_codes.size))
        cols.extend(nbr_codes.tolist())

    if not rows:
        return None

    M = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(coords.shape[0], int(gene_index.height)),
    ).tocsr()
    score, occur = _binary_jaccard_from_csr(M)
    return {
        "score": score,
        "occur": occur,
        "gene_names": [str(g) for g in gene_index.get_column(feature_column).to_list()],
        "transcripts_used": int(coords.shape[0]),
    }


def compute_resolvi_contamination_fast(
    assigned_tx: pl.DataFrame,
    *,
    scrna_reference_path: Optional[Path],
    scrna_celltype_column: str = "cell_type",
    cell_id_column: str = "segger_cell_id",
    feature_column: str = "feature_name",
    x_column: str = "x",
    y_column: str = "y",
    min_transcripts_per_cell: int = 20,
    max_cells: int = 3000,
    k_neighbors: int = 10,
    max_neighbor_distance: float = 20.0,
    alpha_self: float = 0.8,
    alpha_neighbor: float = 0.175,
    alpha_background: float = 0.025,
    contam_cutoff: float = 0.5,
    seed: int = 0,
) -> dict[str, float]:
    """Fast RESOLVI-style contamination estimate (lower is better).

    This approximates the RESOLVI neighborhood contamination formulation on a
    sampled subset of segmented cells by:
    1) deriving host cell types from scRNA reference profile similarity,
    2) mixing self/neighbor/background expected expression per gene,
    3) flagging counts as contaminated when q_self < contam_cutoff.
    """
    out = _empty_resolvi_metrics()
    if scrna_reference_path is None:
        return out
    if alpha_self < 0 or alpha_neighbor < 0 or alpha_background < 0:
        return out
    if alpha_self + alpha_neighbor + alpha_background <= 0:
        return out

    try:
        prepared = _prepare_reference_alignment_fast(
            assigned_tx,
            scrna_reference_path=Path(scrna_reference_path),
            scrna_celltype_column=scrna_celltype_column,
            cell_id_column=cell_id_column,
            feature_column=feature_column,
            x_column=x_column,
            y_column=y_column,
            min_transcripts_per_cell=min_transcripts_per_cell,
            max_cells=max_cells,
            seed=seed,
        )
        if prepared is None:
            return out

        X = prepared["X"]
        centroids = prepared["centroids"]
        cell_weights = prepared["cell_weights"]
        totals = prepared["totals"]
        ref_profiles = prepared["ref_profiles"]
        type_names = prepared["type_names"]
        host_type = prepared["host_type"]

        n_cells = int(X.shape[0])
        n_types = int(ref_profiles.shape[0])
        out["resolvi_shared_genes_used"] = int(X.shape[1])
        out["resolvi_cell_types_used"] = int(len(type_names))
        if n_cells == 0 or n_types == 0:
            return out

        eps = 1e-9
        ref = np.asarray(ref_profiles, dtype=np.float64)

        neighbor_freq = np.zeros((n_cells, n_types), dtype=np.float64)
        if n_cells > 1 and int(k_neighbors) > 0:
            k = min(int(k_neighbors) + 1, n_cells)
            tree = cKDTree(centroids)
            dists, idxs = tree.query(centroids, k=k)
            if k > 1:
                if dists.ndim == 1:
                    dists = dists[:, None]
                    idxs = idxs[:, None]
                nbr_d = dists[:, 1:]
                nbr_i = idxs[:, 1:]
                for i in range(n_cells):
                    if nbr_i.shape[1] == 0:
                        continue
                    if np.isfinite(max_neighbor_distance) and max_neighbor_distance > 0:
                        valid_nbr = nbr_d[i] <= float(max_neighbor_distance)
                    else:
                        valid_nbr = np.ones(nbr_i.shape[1], dtype=bool)
                    pick = nbr_i[i][valid_nbr]
                    if pick.size == 0:
                        continue
                    np.add.at(neighbor_freq[i], host_type[pick], 1.0)
                    s = float(neighbor_freq[i].sum())
                    if s > 0:
                        neighbor_freq[i] /= s

        bg = np.bincount(
            host_type,
            weights=np.asarray(cell_weights, dtype=np.float64),
            minlength=n_types,
        ).astype(np.float64, copy=False)
        bsum = float(bg.sum())
        if bsum > 0:
            bg /= bsum
        p_back = bg @ ref

        per_cell_pct = np.full(n_cells, np.nan, dtype=np.float64)
        per_cell_flag = np.zeros(n_cells, dtype=np.float64)

        for i in range(n_cells):
            row = X.getrow(i)
            if row.nnz == 0:
                continue
            h = int(host_type[i])
            neigh = neighbor_freq[i].copy()
            if 0 <= h < n_types:
                neigh[h] = 0.0
            nsum = float(neigh.sum())
            if nsum > 0:
                neigh /= nsum
            p_self = ref[h]
            p_neigh = neigh @ ref
            denom = (alpha_self * p_self) + (alpha_neighbor * p_neigh) + (alpha_background * p_back) + eps
            q_self = (alpha_self * p_self) / denom
            q_self = np.clip(q_self, 0.0, 1.0)

            vals = row.data.astype(np.float64, copy=False)
            cols = row.indices
            total = float(vals.sum())
            if total <= 0:
                continue
            contam = float(vals[q_self[cols] < float(contam_cutoff)].sum())
            pct = 100.0 * contam / total
            per_cell_pct[i] = pct
            per_cell_flag[i] = 1.0 if contam > 0 else 0.0

        valid_cells = np.isfinite(per_cell_pct) & np.isfinite(totals) & (totals > 0)
        if not np.any(valid_cells):
            return out

        w = totals[valid_cells]
        vals = per_cell_pct[valid_cells]
        flags = per_cell_flag[valid_cells]

        out["resolvi_metric_cells_used"] = int(np.sum(valid_cells))
        out["resolvi_contamination_pct_fast"] = float(np.average(vals, weights=w))
        out["resolvi_contamination_ci95_fast"] = float(_weighted_mean_ci95(vals, w))
        out["resolvi_contaminated_cells_pct_fast"] = float(100.0 * np.average(flags, weights=w))
        out["resolvi_contaminated_cells_pct_ci95_fast"] = float(100.0 * _weighted_bernoulli_ci95(flags, w))
        return out
    except Exception:
        return out


def compute_positive_marker_recall_fast(
    assigned_tx: pl.DataFrame,
    *,
    scrna_reference_path: Optional[Path],
    scrna_celltype_column: str = "cell_type",
    cell_id_column: str = "segger_cell_id",
    feature_column: str = "feature_name",
    x_column: str = "x",
    y_column: str = "y",
    min_transcripts_per_cell: int = 20,
    max_cells: int = 3000,
    n_markers_per_type: int = 12,
    min_specificity_ratio: float = 1.5,
    seed: int = 0,
) -> dict[str, float]:
    """Fast positive-marker recall using inferred host cell types from scRNA."""
    out = {
        "positive_marker_recall_fast": float("nan"),
        "positive_marker_recall_ci95_fast": float("nan"),
        "positive_marker_types_used_fast": 0,
        "positive_marker_genes_used_fast": 0,
        "positive_marker_cells_used_fast": 0,
    }
    prepared = _prepare_reference_alignment_fast(
        assigned_tx,
        scrna_reference_path=scrna_reference_path,
        scrna_celltype_column=scrna_celltype_column,
        cell_id_column=cell_id_column,
        feature_column=feature_column,
        x_column=x_column,
        y_column=y_column,
        min_transcripts_per_cell=min_transcripts_per_cell,
        max_cells=max_cells,
        seed=seed,
    )
    if prepared is None:
        return out

    X = prepared["X"]
    totals = np.asarray(prepared["totals"], dtype=np.float64)
    ref_profiles = np.asarray(prepared["ref_profiles"], dtype=np.float64)
    host_type = np.asarray(prepared["host_type"], dtype=np.int64)

    markers_by_type = _discover_unique_markers_from_profiles(
        ref_profiles,
        n_markers_per_type=n_markers_per_type,
        min_specificity_ratio=min_specificity_ratio,
    )
    if not markers_by_type:
        return out

    out["positive_marker_types_used_fast"] = int(sum(m.size > 0 for m in markers_by_type))
    if out["positive_marker_types_used_fast"] == 0:
        return out
    out["positive_marker_genes_used_fast"] = int(
        len({int(i) for markers in markers_by_type for i in markers.tolist()})
    )

    vals = np.full(X.shape[0], np.nan, dtype=np.float64)
    for i in range(X.shape[0]):
        t = int(host_type[i])
        if t < 0 or t >= len(markers_by_type):
            continue
        marker_idx = markers_by_type[t]
        if marker_idx.size == 0:
            continue
        marker_weights = ref_profiles[t, marker_idx]
        denom = float(np.sum(marker_weights))
        if denom <= 0:
            continue
        row = X.getrow(i)
        marker_vals = np.asarray(row[:, marker_idx].toarray()).ravel()
        vals[i] = 100.0 * float(np.sum(marker_weights[marker_vals > 0])) / denom

    valid = np.isfinite(vals) & np.isfinite(totals) & (totals > 0)
    if not np.any(valid):
        return out

    w = totals[valid]
    v = vals[valid]
    out["positive_marker_cells_used_fast"] = int(np.sum(valid))
    out["positive_marker_recall_fast"] = float(np.average(v, weights=w))
    out["positive_marker_recall_ci95_fast"] = float(_weighted_mean_ci95(v, w))
    return out


def compute_spurious_coexpression_fast(
    source_tx: pl.DataFrame,
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    reference_cell_id_column: str = "cell_id",
    feature_column: str = "feature_name",
    x_column: str = "x",
    y_column: str = "y",
    compartment_column: str = "cell_compartment",
    nucleus_value: int = 2,
    min_transcripts_per_cell: int = 20,
    max_cells: int = 3000,
    spatial_radius: float = 10.0,
    max_spatial_transcripts: int = 10000,
    min_gene_count: int = 20,
    min_support: int = 20,
    ratio_cutoff: float = 2.0,
    nuclear_max: float = 1e-3,
    min_spatial_score: float = 1e-3,
    max_pairs: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Fast nucleus-aware spurious co-expression score (lower is better)."""
    out = {
        "spurious_coexpression_fast": float("nan"),
        "spurious_coexpression_ci95_fast": float("nan"),
        "spurious_pairs_used_fast": 0,
        "spurious_pairs_discovered_fast": 0,
        "spurious_source_transcripts_used_fast": 0,
    }
    spatial = _spatial_gene_scores_fast(
        source_tx,
        feature_column=feature_column,
        x_column=x_column,
        y_column=y_column,
        radius=spatial_radius,
        max_transcripts=max_spatial_transcripts,
        min_gene_count=min_gene_count,
        seed=seed,
    )
    if spatial is None:
        return out

    spatial_score = np.asarray(spatial["score"], dtype=np.float64)
    spatial_occur = np.asarray(spatial["occur"], dtype=np.float64)
    spatial_gene_names = [str(g) for g in spatial["gene_names"]]
    out["spurious_source_transcripts_used_fast"] = int(spatial["transcripts_used"])
    if spatial_score.shape[0] < 2:
        return out

    req_source = [reference_cell_id_column, feature_column, compartment_column]
    if any(col not in source_tx.columns for col in req_source):
        return out

    gene_index = pl.DataFrame({feature_column: spatial_gene_names}).with_row_index(name="_gid")
    nuc_df = (
        source_tx.select(req_source)
        .drop_nulls()
        .with_columns(
            pl.col(reference_cell_id_column).cast(pl.Utf8),
            pl.col(feature_column).cast(pl.Utf8),
        )
        .filter(valid_cell_id_expr(reference_cell_id_column))
        .filter(pl.col(compartment_column) == nucleus_value)
        .join(gene_index, on=feature_column, how="inner")
    )
    if nuc_df.height == 0:
        return out

    nuc_cells = (
        nuc_df.select(reference_cell_id_column)
        .unique()
        .sort(reference_cell_id_column)
        .with_row_index(name="_cid")
    )
    nuc_mapped = nuc_df.join(nuc_cells, on=reference_cell_id_column, how="inner")
    nuc_counts = nuc_mapped.group_by(["_cid", "_gid"]).agg(pl.len().alias("_count"))
    if nuc_counts.height == 0:
        return out

    nuc_rows = nuc_counts.get_column("_cid").to_numpy().astype(np.int64, copy=False)
    nuc_cols = nuc_counts.get_column("_gid").to_numpy().astype(np.int64, copy=False)
    nuc_data = np.ones(nuc_counts.height, dtype=np.float64)
    X_nuc = sparse.coo_matrix(
        (nuc_data, (nuc_rows, nuc_cols)),
        shape=(int(nuc_cells.height), len(spatial_gene_names)),
    ).tocsr()
    nuclear_score, _ = _binary_jaccard_from_csr(X_nuc)
    if nuclear_score.shape != spatial_score.shape:
        return out

    support = spatial_occur >= float(min_support)
    ratio = spatial_score / (nuclear_score + 1e-12)
    mask = (
        (ratio > float(ratio_cutoff))
        & (nuclear_score < float(nuclear_max))
        & (spatial_score > float(min_spatial_score))
        & support[:, None]
        & support[None, :]
    )
    np.fill_diagonal(mask, False)
    pair_i, pair_j = np.where(np.triu(mask, 1))
    if pair_i.size == 0:
        return out

    pair_ratio = ratio[pair_i, pair_j]
    order = np.argsort(pair_ratio)[::-1]
    if max_pairs > 0 and order.size > int(max_pairs):
        order = order[: int(max_pairs)]
    pair_i = pair_i[order]
    pair_j = pair_j[order]
    out["spurious_pairs_discovered_fast"] = int(pair_i.size)

    built = _build_cell_gene_matrix(
        assigned_tx,
        cell_id_column=cell_id_column,
        feature_column=feature_column,
        x_column=x_column,
        y_column=y_column,
        min_transcripts_per_cell=min_transcripts_per_cell,
        max_cells=max_cells,
        seed=seed,
    )
    if built is None:
        return out

    X_seg, _, _, seg_gene_names = built
    seg_gene_to_idx = {str(g): i for i, g in enumerate(seg_gene_names)}
    seg_i: list[int] = []
    seg_j: list[int] = []
    nuc_vals: list[float] = []
    weights: list[float] = []
    for gi, gj in zip(pair_i.tolist(), pair_j.tolist()):
        name_i = spatial_gene_names[int(gi)]
        name_j = spatial_gene_names[int(gj)]
        si = seg_gene_to_idx.get(name_i)
        sj = seg_gene_to_idx.get(name_j)
        if si is None or sj is None:
            continue
        seg_i.append(int(si))
        seg_j.append(int(sj))
        nuc_vals.append(float(nuclear_score[int(gi), int(gj)]))
        weights.append(float(np.log1p(max(spatial_occur[int(gi)], spatial_occur[int(gj)]))))

    if not seg_i:
        return out

    Xi = X_seg[:, np.asarray(seg_i, dtype=np.int64)]
    Xj = X_seg[:, np.asarray(seg_j, dtype=np.int64)]
    co = Xi.minimum(Xj).sum(axis=0).A1.astype(np.float64, copy=False)
    union = Xi.maximum(Xj).sum(axis=0).A1.astype(np.float64, copy=False)
    score_seg = co / (union + 1e-12)
    score_nuc = np.asarray(nuc_vals, dtype=np.float64)
    excess = np.maximum(score_seg - score_nuc, 0.0)
    pair_weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(excess) & np.isfinite(pair_weights) & (pair_weights > 0)
    if not np.any(valid):
        return out

    out["spurious_pairs_used_fast"] = int(np.sum(valid))
    out["spurious_coexpression_fast"] = float(np.average(excess[valid], weights=pair_weights[valid]))
    out["spurious_coexpression_ci95_fast"] = float(
        _weighted_mean_ci95(excess[valid], pair_weights[valid])
    )
    return out


def compute_center_border_ncv_fast(
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    feature_column: str = "feature_name",
    x_column: str = "x",
    y_column: str = "y",
    erosion_fraction: float = 0.3,
    min_transcripts_per_cell: int = 20,
    max_cells: int = 3000,
    n_neighbors: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Fast center-border NCV score (higher is better)."""
    out = {
        "center_border_ncv_score_fast": float("nan"),
        "center_border_ncv_ci95_fast": float("nan"),
        "center_border_ncv_ratio_fast": float("nan"),
        "center_border_ncv_cells_used_fast": 0,
    }
    req = [cell_id_column, feature_column, x_column, y_column]
    if any(col not in assigned_tx.columns for col in req):
        return out

    df = (
        assigned_tx.select(req)
        .drop_nulls()
        .with_columns(
            pl.col(cell_id_column).cast(pl.Utf8),
            pl.col(feature_column).cast(pl.Utf8),
        )
        .filter(valid_cell_id_expr(cell_id_column))
    )
    if df.height == 0:
        return out

    cell_stats = (
        df.group_by(cell_id_column)
        .agg(
            pl.len().alias("n_total"),
            pl.col(x_column).mean().alias("cx"),
            pl.col(y_column).mean().alias("cy"),
            pl.col(x_column).min().alias("x_min"),
            pl.col(x_column).max().alias("x_max"),
            pl.col(y_column).min().alias("y_min"),
            pl.col(y_column).max().alias("y_max"),
        )
        .with_columns(
            (pl.col("x_max") - pl.col("x_min")).alias("width"),
            (pl.col("y_max") - pl.col("y_min")).alias("height"),
        )
        .with_columns(
            (pl.min_horizontal("width", "height") * float(erosion_fraction)).alias("erosion"),
        )
        .filter(pl.col("n_total") >= int(min_transcripts_per_cell))
        .filter((pl.col("width") > 0) & (pl.col("height") > 0))
        .with_columns(
            (pl.col("x_min") + pl.col("erosion")).alias("cx_min"),
            (pl.col("x_max") - pl.col("erosion")).alias("cx_max"),
            (pl.col("y_min") + pl.col("erosion")).alias("cy_min"),
            (pl.col("y_max") - pl.col("erosion")).alias("cy_max"),
        )
        .filter((pl.col("cx_max") > pl.col("cx_min")) & (pl.col("cy_max") > pl.col("cy_min")))
    )
    if cell_stats.height == 0:
        return out

    if max_cells > 0 and cell_stats.height > max_cells:
        ids = np.asarray(cell_stats.get_column(cell_id_column).to_list(), dtype=object)
        rng = np.random.default_rng(seed)
        picked = rng.choice(ids, size=int(max_cells), replace=False).tolist()
        cell_stats = cell_stats.filter(pl.col(cell_id_column).is_in(picked))

    df = df.join(
        cell_stats.select(
            [cell_id_column, "n_total", "cx", "cy", "cx_min", "cx_max", "cy_min", "cy_max"]
        ),
        on=cell_id_column,
        how="inner",
    )
    if df.height == 0:
        return out

    classified = df.with_columns(
        (
            (pl.col(x_column) >= pl.col("cx_min"))
            & (pl.col(x_column) <= pl.col("cx_max"))
            & (pl.col(y_column) >= pl.col("cy_min"))
            & (pl.col(y_column) <= pl.col("cy_max"))
        ).alias("is_center")
    )

    center_df = (
        classified.filter(pl.col("is_center"))
        .group_by([cell_id_column, feature_column])
        .agg(pl.len().alias("count"))
    )
    border_df = (
        classified.filter(~pl.col("is_center"))
        .group_by([cell_id_column, feature_column])
        .agg(pl.len().alias("count"))
    )
    full_df = df.group_by([cell_id_column, feature_column]).agg(pl.len().alias("count"))
    if center_df.height == 0 or border_df.height == 0 or full_df.height == 0:
        return out

    def _to_expr_dict(expr_df: pl.DataFrame) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for row in expr_df.iter_rows(named=True):
            cid = str(row[cell_id_column])
            gene = str(row[feature_column])
            result.setdefault(cid, {})[gene] = float(row["count"])
        return result

    center_expr = _to_expr_dict(center_df)
    border_expr = _to_expr_dict(border_df)
    full_expr = _to_expr_dict(full_df)
    if not center_expr or not border_expr or not full_expr:
        return out

    ids = [str(v) for v in cell_stats.get_column(cell_id_column).to_list()]
    coords = cell_stats.select(["cx", "cy"]).to_numpy().astype(np.float64, copy=False)
    n_total = cell_stats.get_column("n_total").to_numpy().astype(np.float64, copy=False)
    if coords.shape[0] < 2:
        return out

    tree = cKDTree(coords)
    k = min(int(n_neighbors) + 1, coords.shape[0])
    dists, idxs = tree.query(coords, k=k)
    if k <= 1:
        return out
    if dists.ndim == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    scores: list[float] = []
    ratios: list[float] = []
    weights: list[float] = []
    for i, cid in enumerate(ids):
        center = center_expr.get(cid)
        border = border_expr.get(cid)
        if not center or not border:
            continue

        nbr_ids = [ids[int(j)] for j in idxs[i, 1:] if int(j) != i]
        if not nbr_ids:
            continue

        neighbor_acc: dict[str, float] = {}
        for nbr in nbr_ids:
            expr = full_expr.get(nbr)
            if not expr:
                continue
            for gene, count in expr.items():
                neighbor_acc[gene] = neighbor_acc.get(gene, 0.0) + float(count)
        if not neighbor_acc:
            continue

        scale = float(len(nbr_ids))
        neighbor = {g: v / scale for g, v in neighbor_acc.items()}
        genes = sorted(set(center) | set(border) | set(neighbor))
        if len(genes) < 5:
            continue

        center_vec = np.asarray([center.get(g, 0.0) for g in genes], dtype=np.float64)
        border_vec = np.asarray([border.get(g, 0.0) for g in genes], dtype=np.float64)
        neighbor_vec = np.asarray([neighbor.get(g, 0.0) for g in genes], dtype=np.float64)
        sim_center_border = _cosine_similarity(center_vec, border_vec)
        sim_border_ncv = _cosine_similarity(border_vec, neighbor_vec)
        if not np.isfinite(sim_center_border) or not np.isfinite(sim_border_ncv):
            continue

        if sim_center_border > 0.01:
            ratio = float(sim_border_ncv / sim_center_border)
        else:
            ratio = 1.0
        score = 1.0 / (1.0 + max(0.0, ratio - 1.0))
        scores.append(float(score))
        ratios.append(float(ratio))
        weights.append(float(n_total[i]))

    if not scores:
        return out

    v = np.asarray(scores, dtype=np.float64)
    r = np.asarray(ratios, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    out["center_border_ncv_cells_used_fast"] = int(v.size)
    out["center_border_ncv_score_fast"] = float(np.average(v, weights=w))
    out["center_border_ncv_ci95_fast"] = float(_weighted_mean_ci95(v, w))
    out["center_border_ncv_ratio_fast"] = float(np.average(r, weights=w))
    return out


def compute_reference_morphology_match_fast(
    source_tx: pl.DataFrame,
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    reference_cell_id_column: str = "cell_id",
    x_column: str = "x",
    y_column: str = "y",
) -> dict[str, float]:
    """Fast matched-cell morphology agreement against source reference cells."""
    out = {
        "reference_morphology_match_fast": float("nan"),
        "reference_morphology_match_ci95_fast": float("nan"),
        "reference_morphology_cells_used_fast": 0,
    }
    if reference_cell_id_column not in source_tx.columns or reference_cell_id_column not in assigned_tx.columns:
        return out

    pred_geom = _simple_geometry_table(
        assigned_tx,
        cell_id_column=cell_id_column,
        x_column=x_column,
        y_column=y_column,
    )
    ref_geom = _simple_geometry_table(
        source_tx,
        cell_id_column=reference_cell_id_column,
        x_column=x_column,
        y_column=y_column,
    )
    if pred_geom.height == 0 or ref_geom.height == 0:
        return out

    overlap = (
        assigned_tx.select([cell_id_column, reference_cell_id_column])
        .drop_nulls()
        .with_columns(
            pl.col(cell_id_column).cast(pl.Utf8),
            pl.col(reference_cell_id_column).cast(pl.Utf8),
        )
        .filter(valid_cell_id_expr(cell_id_column))
        .filter(valid_cell_id_expr(reference_cell_id_column))
        .group_by([cell_id_column, reference_cell_id_column])
        .agg(pl.len().alias("overlap_count"))
    )
    if overlap.height == 0:
        return out

    mapping = (
        overlap.sort(
            by=[cell_id_column, "overlap_count", reference_cell_id_column],
            descending=[False, True, False],
        )
        .group_by(cell_id_column)
        .agg(
            pl.first(reference_cell_id_column).alias(reference_cell_id_column),
            pl.first("overlap_count").alias("overlap_count"),
        )
    )
    if mapping.height == 0:
        return out

    ref_geom_prefixed = ref_geom.rename(
        {
            reference_cell_id_column: "reference_match_id",
            "area": "reference_area",
            "elongation": "reference_elongation",
            "circularity": "reference_circularity",
        }
    )
    joined = (
        mapping.rename({reference_cell_id_column: "reference_match_id"})
        .join(pred_geom, on=cell_id_column, how="inner")
        .join(ref_geom_prefixed, on="reference_match_id", how="inner")
    )
    if joined.height == 0:
        return out

    pred_area = joined.get_column("area").to_numpy().astype(np.float64, copy=False)
    pred_elong = joined.get_column("elongation").to_numpy().astype(np.float64, copy=False)
    pred_circ = joined.get_column("circularity").to_numpy().astype(np.float64, copy=False)
    ref_area = joined.get_column("reference_area").to_numpy().astype(np.float64, copy=False)
    ref_elong = joined.get_column("reference_elongation").to_numpy().astype(np.float64, copy=False)
    ref_circ = joined.get_column("reference_circularity").to_numpy().astype(np.float64, copy=False)
    w = joined.get_column("overlap_count").to_numpy().astype(np.float64, copy=False)

    area_sim = np.exp(-np.abs(np.log((pred_area + 1e-9) / (ref_area + 1e-9))))
    elong_sim = np.exp(-np.abs(np.log((pred_elong + 1e-9) / (ref_elong + 1e-9))))
    circ_sim = np.exp(-np.abs(pred_circ - ref_circ))
    score = (area_sim + elong_sim + circ_sim) / 3.0

    valid = np.isfinite(score) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return out

    out["reference_morphology_cells_used_fast"] = int(np.sum(valid))
    out["reference_morphology_match_fast"] = float(np.average(score[valid], weights=w[valid]))
    out["reference_morphology_match_ci95_fast"] = float(_weighted_mean_ci95(score[valid], w[valid]))
    return out


def compute_border_contamination_fast(
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    x_column: str = "x",
    y_column: str = "y",
    erosion_fraction: float = 0.3,
    min_transcripts_per_cell: int = 20,
    max_cells: int = 3000,
    contaminated_enrichment_threshold: float = 1.25,
    seed: int = 0,
) -> dict[str, float]:
    """Fast border-enrichment contamination proxy (lower is better).

    This approximates periphery contamination by comparing transcript density
    in border vs center regions defined by an eroded bounding box.
    """
    eps = 1e-9
    req = [cell_id_column, x_column, y_column]
    for col in req:
        if col not in assigned_tx.columns:
            return {
                "border_contamination_fast": float("nan"),
                "border_enrichment_fast": float("nan"),
                "border_excess_pct_fast": float("nan"),
                "border_contaminated_cells_pct_fast": float("nan"),
                "border_contaminated_cells_pct_ci95_fast": float("nan"),
                "border_metric_cells_used": 0,
            }

    df = assigned_tx.select(req).drop_nulls()
    if df.height == 0:
        return {
            "border_contamination_fast": float("nan"),
            "border_enrichment_fast": float("nan"),
            "border_excess_pct_fast": float("nan"),
            "border_contaminated_cells_pct_fast": float("nan"),
            "border_contaminated_cells_pct_ci95_fast": float("nan"),
            "border_metric_cells_used": 0,
        }

    cell_stats = (
        df.group_by(cell_id_column)
        .agg(
            pl.len().alias("n_total"),
            pl.col(x_column).min().alias("x_min"),
            pl.col(x_column).max().alias("x_max"),
            pl.col(y_column).min().alias("y_min"),
            pl.col(y_column).max().alias("y_max"),
        )
        .with_columns(
            (pl.col("x_max") - pl.col("x_min")).alias("width"),
            (pl.col("y_max") - pl.col("y_min")).alias("height"),
        )
        .with_columns(
            pl.min_horizontal("width", "height").alias("min_side"),
            (pl.min_horizontal("width", "height") * erosion_fraction).alias("erosion"),
        )
        .filter(pl.col("n_total") >= min_transcripts_per_cell)
        .filter((pl.col("width") > 0) & (pl.col("height") > 0))
        .filter((pl.col("min_side") > 0) & (pl.col("erosion") > 0))
        .with_columns(
            (pl.col("x_min") + pl.col("erosion")).alias("cx_min"),
            (pl.col("x_max") - pl.col("erosion")).alias("cx_max"),
            (pl.col("y_min") + pl.col("erosion")).alias("cy_min"),
            (pl.col("y_max") - pl.col("erosion")).alias("cy_max"),
        )
        .filter((pl.col("cx_max") > pl.col("cx_min")) & (pl.col("cy_max") > pl.col("cy_min")))
    )

    if cell_stats.height == 0:
        return {
            "border_contamination_fast": float("nan"),
            "border_enrichment_fast": float("nan"),
            "border_excess_pct_fast": float("nan"),
            "border_contaminated_cells_pct_fast": float("nan"),
            "border_contaminated_cells_pct_ci95_fast": float("nan"),
            "border_metric_cells_used": 0,
        }

    if max_cells > 0 and cell_stats.height > max_cells:
        rng = np.random.default_rng(seed)
        cell_ids = np.array(cell_stats.get_column(cell_id_column).to_list(), dtype=object)
        picked = rng.choice(cell_ids, size=max_cells, replace=False).tolist()
        cell_stats = cell_stats.filter(pl.col(cell_id_column).is_in(picked))
        df = df.join(cell_stats.select([cell_id_column]), on=cell_id_column, how="inner")

    classified = (
        df.join(
            cell_stats.select(
                [
                    cell_id_column,
                    "cx_min",
                    "cx_max",
                    "cy_min",
                    "cy_max",
                    "width",
                    "height",
                    "n_total",
                ]
            ),
            on=cell_id_column,
            how="inner",
        )
        .with_columns(
            (
                (pl.col(x_column) >= pl.col("cx_min"))
                & (pl.col(x_column) <= pl.col("cx_max"))
                & (pl.col(y_column) >= pl.col("cy_min"))
                & (pl.col(y_column) <= pl.col("cy_max"))
            ).alias("is_center")
        )
    )

    grouped = (
        classified.group_by([cell_id_column, "is_center"])
        .agg(pl.len().alias("n"))
    )
    n_center = grouped.filter(pl.col("is_center")).select([cell_id_column, pl.col("n").alias("n_center")])
    n_border = grouped.filter(~pl.col("is_center")).select([cell_id_column, pl.col("n").alias("n_border")])

    per_cell = (
        cell_stats.join(n_center, on=cell_id_column, how="left")
        .join(n_border, on=cell_id_column, how="left")
        .with_columns(
            pl.col("n_center").fill_null(0).cast(pl.Float64),
            pl.col("n_border").fill_null(0).cast(pl.Float64),
            pl.col("n_total").cast(pl.Float64),
        )
        .with_columns(
            (pl.col("width") * pl.col("height")).alias("bbox_area"),
            ((pl.col("cx_max") - pl.col("cx_min")) * (pl.col("cy_max") - pl.col("cy_min"))).alias("center_area"),
        )
        .with_columns(
            (pl.col("bbox_area") - pl.col("center_area")).alias("border_area"),
        )
        .with_columns(
            (pl.col("n_center") / pl.max_horizontal(pl.col("center_area"), pl.lit(eps))).alias("center_density"),
            (pl.col("n_border") / pl.max_horizontal(pl.col("border_area"), pl.lit(eps))).alias("border_density"),
        )
        .with_columns(
            (pl.col("border_density") / pl.max_horizontal(pl.col("center_density"), pl.lit(eps))).alias("border_enrichment"),
            (
                pl.when(
                    (pl.col("border_density") / pl.max_horizontal(pl.col("center_density"), pl.lit(eps)) - 1.0)
                    > 0
                )
                .then(pl.col("border_density") / pl.max_horizontal(pl.col("center_density"), pl.lit(eps)) - 1.0)
                .otherwise(0.0)
            ).alias("contam_score"),
        )
    )

    if per_cell.height == 0:
        return {
            "border_contamination_fast": float("nan"),
            "border_enrichment_fast": float("nan"),
            "border_excess_pct_fast": float("nan"),
            "border_contaminated_cells_pct_fast": float("nan"),
            "border_contaminated_cells_pct_ci95_fast": float("nan"),
            "border_metric_cells_used": 0,
        }

    weights = per_cell.get_column("n_total").to_numpy()
    contam = np.average(per_cell.get_column("contam_score").to_numpy(), weights=weights)
    enrich = np.average(per_cell.get_column("border_enrichment").to_numpy(), weights=weights)
    border_excess_pct = max(0.0, (float(enrich) - 1.0) * 100.0)
    contaminated_flags = (
        per_cell.get_column("border_enrichment").to_numpy()
        > float(contaminated_enrichment_threshold)
    ).astype(np.float64)
    contaminated_cells_pct = 100.0 * np.average(
        contaminated_flags,
        weights=weights,
    )
    contaminated_cells_pct_ci95 = 100.0 * _weighted_bernoulli_ci95(
        contaminated_flags,
        weights,
    )
    return {
        "border_contamination_fast": float(contam),
        "border_enrichment_fast": float(enrich),
        "border_excess_pct_fast": float(border_excess_pct),
        "border_contaminated_cells_pct_fast": float(contaminated_cells_pct),
        "border_contaminated_cells_pct_ci95_fast": float(contaminated_cells_pct_ci95),
        "border_metric_cells_used": int(per_cell.height),
    }


def compute_transcript_centroid_offset_fast(
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    x_column: str = "x",
    y_column: str = "y",
    min_transcripts_per_cell: int = 20,
    max_cells: int = 3000,
    seed: int = 0,
) -> dict[str, float]:
    """Fast transcript centroid-offset metric (higher is better).

    Uses a bounding-box center as cell centroid approximation.
    """
    req = [cell_id_column, x_column, y_column]
    for col in req:
        if col not in assigned_tx.columns:
            return {
                "transcript_centroid_offset_fast": float("nan"),
                "transcript_centroid_offset_ci95_fast": float("nan"),
                "tco_metric_cells_used": 0,
            }

    stats = (
        assigned_tx.select(req)
        .drop_nulls()
        .group_by(cell_id_column)
        .agg(
            pl.len().alias("n_total"),
            pl.col(x_column).mean().alias("tx_cx"),
            pl.col(y_column).mean().alias("tx_cy"),
            pl.col(x_column).min().alias("x_min"),
            pl.col(x_column).max().alias("x_max"),
            pl.col(y_column).min().alias("y_min"),
            pl.col(y_column).max().alias("y_max"),
        )
        .filter(pl.col("n_total") >= min_transcripts_per_cell)
        .with_columns(
            (pl.col("x_max") - pl.col("x_min")).alias("width"),
            (pl.col("y_max") - pl.col("y_min")).alias("height"),
        )
        .filter((pl.col("width") > 0) & (pl.col("height") > 0))
        .with_columns(
            ((pl.col("x_min") + pl.col("x_max")) / 2.0).alias("cell_cx"),
            ((pl.col("y_min") + pl.col("y_max")) / 2.0).alias("cell_cy"),
            (pl.col("width") * pl.col("height")).alias("area"),
        )
        .filter(pl.col("area") > 0)
    )

    if stats.height == 0:
        return {
            "transcript_centroid_offset_fast": float("nan"),
            "transcript_centroid_offset_ci95_fast": float("nan"),
            "tco_metric_cells_used": 0,
        }

    if max_cells > 0 and stats.height > max_cells:
        rng = np.random.default_rng(seed)
        ids = np.array(stats.get_column(cell_id_column).to_list(), dtype=object)
        picked = rng.choice(ids, size=max_cells, replace=False).tolist()
        stats = stats.filter(pl.col(cell_id_column).is_in(picked))

    stats = stats.with_columns(
        (
            (
                (pl.col("tx_cx") - pl.col("cell_cx")) ** 2
                + (pl.col("tx_cy") - pl.col("cell_cy")) ** 2
            ).sqrt()
        ).alias("centroid_offset")
    ).with_columns(
        (
            1.0 - (pl.col("centroid_offset") / (pl.col("area").sqrt() + 1e-9))
        ).clip(lower_bound=0.0, upper_bound=1.0).alias("tco_score")
    )

    weights = stats.get_column("n_total").to_numpy().astype(np.float64, copy=False)
    tco_vals = stats.get_column("tco_score").to_numpy().astype(np.float64, copy=False)
    tco = float(np.average(tco_vals, weights=weights))
    tco_ci95 = _weighted_mean_ci95(tco_vals, weights)
    return {
        "transcript_centroid_offset_fast": tco,
        "transcript_centroid_offset_ci95_fast": float(tco_ci95),
        "tco_metric_cells_used": int(stats.height),
    }


def compute_signal_doublet_fast(
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    z_column: str = "z",
    min_transcripts_per_cell: int = 20,
    max_cells: int = 3000,
    seed: int = 0,
    doublet_threshold: float = 0.6,
) -> dict[str, float]:
    """Fast 3D doublet-like fraction based on per-cell z-spread."""
    if z_column not in assigned_tx.columns or cell_id_column not in assigned_tx.columns:
        return {
            "signal_doublet_like_fraction_fast": float("nan"),
            "signal_doublet_like_fraction_ci95_fast": float("nan"),
            "signal_metric_cells_used": 0,
        }

    stats = (
        assigned_tx.select([cell_id_column, z_column])
        .drop_nulls()
        .group_by(cell_id_column)
        .agg(
            pl.len().alias("n_total"),
            pl.col(z_column).std().alias("z_std"),
        )
        .filter(pl.col("n_total") >= min_transcripts_per_cell)
        .drop_nulls(["z_std"])
    )

    if stats.height == 0:
        return {
            "signal_doublet_like_fraction_fast": float("nan"),
            "signal_doublet_like_fraction_ci95_fast": float("nan"),
            "signal_metric_cells_used": 0,
        }

    if max_cells > 0 and stats.height > max_cells:
        rng = np.random.default_rng(seed)
        ids = np.array(stats.get_column(cell_id_column).to_list(), dtype=object)
        picked = rng.choice(ids, size=max_cells, replace=False).tolist()
        stats = stats.filter(pl.col(cell_id_column).is_in(picked))

    n_total = stats.get_column("n_total").to_numpy().astype(np.float64, copy=False)
    z_std = stats.get_column("z_std").to_numpy().astype(np.float64, copy=False)
    z_std = np.where(np.isfinite(z_std), z_std, np.nan)
    positive = z_std[np.isfinite(z_std) & (z_std > 0)]

    if positive.size == 0:
        ci95 = _weighted_bernoulli_ci95(
            np.zeros_like(n_total, dtype=np.float64),
            n_total,
        )
        return {
            "signal_doublet_like_fraction_fast": 0.0,
            "signal_doublet_like_fraction_ci95_fast": float(ci95),
            "signal_metric_cells_used": int(stats.height),
        }

    expected = float(np.median(positive))
    if expected <= 1e-12:
        ci95 = _weighted_bernoulli_ci95(
            np.zeros_like(n_total, dtype=np.float64),
            n_total,
        )
        return {
            "signal_doublet_like_fraction_fast": 0.0,
            "signal_doublet_like_fraction_ci95_fast": float(ci95),
            "signal_metric_cells_used": int(stats.height),
        }

    integrity = np.clip(expected / (z_std + 1e-9), 0.0, 1.0)
    doublet_flags = (integrity < doublet_threshold).astype(np.float64)
    doublet_like = float(np.average(doublet_flags, weights=n_total))
    doublet_like_ci95 = _weighted_bernoulli_ci95(doublet_flags, n_total)
    return {
        "signal_doublet_like_fraction_fast": doublet_like,
        "signal_doublet_like_fraction_ci95_fast": float(doublet_like_ci95),
        "signal_metric_cells_used": int(stats.height),
    }


def _empty_signal_hotspot_metrics() -> dict[str, float | int]:
    """Default empty return payload for hotspot-restricted vertical metric."""
    return {
        "signal_hotspot_doublet_fraction_fast": float("nan"),
        "signal_hotspot_doublet_fraction_ci95_fast": float("nan"),
        "signal_hotspot_cutoff_fast": float("nan"),
        "signal_hotspot_pixels_used_fast": 0,
        "signal_hotspot_candidate_cells_fast": 0,
        "signal_hotspot_metric_cells_used_fast": 0,
        "signal_hotspot_cells_scored_fast": 0,
    }


def _sorted_value_knee(values: np.ndarray) -> float:
    """Pick a low-tail cutoff using a simple knee on sorted values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 16:
        return float("nan")

    v.sort()
    v_min = float(v[0])
    v_max = float(v[-1])
    if not np.isfinite(v_min) or not np.isfinite(v_max) or (v_max - v_min) < 1e-6:
        return float("nan")

    x = np.linspace(0.0, 1.0, v.size, dtype=np.float64)
    y = (v - v_min) / (v_max - v_min)
    scores = x - y
    if v.size > 2:
        idx = int(np.argmax(scores[1:-1])) + 1
    else:
        idx = int(np.argmax(scores))
    return float(v[idx])


def compute_signal_hotspot_doublet_fast(
    source_tx: pl.DataFrame,
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    feature_column: str = "feature_name",
    x_column: str = "x",
    y_column: str = "y",
    z_column: str = "z",
    grid_size: float = 3.0,
    min_pixel_signal: int = 3,
    min_transcripts_per_cell: int = 20,
    min_side_transcripts: int = 5,
    max_cells: int = 3000,
    seed: int = 0,
    doublet_threshold: float = 0.6,
) -> dict[str, float | int]:
    """Fast hotspot-restricted vertical doublet metric using a global integrity proxy."""
    result = _empty_signal_hotspot_metrics()

    required_source = [feature_column, x_column, y_column, z_column]
    required_assigned = [cell_id_column, feature_column, x_column, y_column, z_column]
    if any(col not in source_tx.columns for col in required_source):
        return result
    if any(col not in assigned_tx.columns for col in required_assigned):
        return result
    if grid_size <= 0:
        return result

    source = source_tx.select(required_source).drop_nulls()
    assigned = assigned_tx.select(required_assigned).drop_nulls()
    if source.height == 0 or assigned.height == 0:
        return result

    mins = source.select(
        pl.col(x_column).min().alias("min_x"),
        pl.col(y_column).min().alias("min_y"),
    )
    min_x = float(mins.item(0, "min_x"))
    min_y = float(mins.item(0, "min_y"))

    z_lookup = (
        source.select(pl.col(z_column).unique().sort())
        .with_row_index("z_index")
        .with_columns(pl.col("z_index").cast(pl.Int32))
    )
    if z_lookup.height < 2:
        return result

    pixel_exprs = [
        (((pl.col(x_column) - min_x) / grid_size).floor().cast(pl.Int32)).alias("x_pixel"),
        (((pl.col(y_column) - min_y) / grid_size).floor().cast(pl.Int32)).alias("y_pixel"),
    ]

    source_binned = source.with_columns(pixel_exprs).join(z_lookup, on=z_column, how="inner")
    assigned_binned = assigned.with_columns(pixel_exprs).join(z_lookup, on=z_column, how="inner")
    if source_binned.height == 0 or assigned_binned.height == 0:
        return result

    cube = (
        source_binned.group_by(["x_pixel", "y_pixel", "z_index", feature_column])
        .agg(pl.len().alias("count"))
        .with_columns(pl.col("count").cast(pl.Int32))
    )
    if cube.height == 0:
        return result

    plane_stats = (
        cube.group_by(["x_pixel", "y_pixel", "z_index"])
        .agg(
            pl.sum("count").alias("n_plane"),
            (pl.col("count") * pl.col("count")).sum().alias("norm_sq"),
        )
        .with_columns(
            pl.col("n_plane").cast(pl.Float64),
            pl.col("norm_sq").cast(pl.Float64),
        )
    )

    lower_planes = plane_stats.rename(
        {"n_plane": "n_plane_lower", "norm_sq": "norm_sq_lower"}
    )
    upper_planes = (
        plane_stats.rename({"n_plane": "n_plane_upper", "norm_sq": "norm_sq_upper"})
        .with_columns((pl.col("z_index") - 1).alias("z_index"))
    )
    plane_pairs = lower_planes.join(
        upper_planes,
        on=["x_pixel", "y_pixel", "z_index"],
        how="inner",
    )
    if plane_pairs.height == 0:
        return result

    lower_gene = cube.rename({"count": "count_lower"})
    upper_gene = cube.rename({"count": "count_upper"}).with_columns(
        (pl.col("z_index") - 1).alias("z_index")
    )
    pair_dot = (
        lower_gene.join(
            upper_gene,
            on=["x_pixel", "y_pixel", "z_index", feature_column],
            how="inner",
        )
        .group_by(["x_pixel", "y_pixel", "z_index"])
        .agg(
            (pl.col("count_lower") * pl.col("count_upper"))
            .sum()
            .cast(pl.Float64)
            .alias("dot")
        )
    )

    pair_stats = (
        plane_pairs.join(pair_dot, on=["x_pixel", "y_pixel", "z_index"], how="left")
        .with_columns(pl.col("dot").fill_null(0.0))
        .with_columns(
            (
                pl.col("dot")
                / (
                    (pl.col("norm_sq_lower").sqrt() * pl.col("norm_sq_upper").sqrt())
                    + 1e-9
                )
            )
            .clip(lower_bound=0.0, upper_bound=1.0)
            .alias("pair_cosine"),
            (pl.col("n_plane_lower") + pl.col("n_plane_upper")).alias("pair_weight"),
        )
    )

    pixel_signal = (
        cube.group_by(["x_pixel", "y_pixel"])
        .agg(pl.sum("count").alias("pixel_signal"))
        .with_columns(pl.col("pixel_signal").cast(pl.Float64))
    )
    pixel_integrity = (
        pair_stats.group_by(["x_pixel", "y_pixel"])
        .agg(
            (pl.col("pair_cosine") * pl.col("pair_weight")).sum().alias("weighted_sum"),
            pl.sum("pair_weight").alias("weight_sum"),
        )
        .join(pixel_signal, on=["x_pixel", "y_pixel"], how="inner")
        .filter(pl.col("pixel_signal") >= float(min_pixel_signal))
        .with_columns(
            (
                pl.col("weighted_sum") / (pl.col("weight_sum") + 1e-9)
            )
            .clip(lower_bound=0.0, upper_bound=1.0)
            .alias("integrity")
        )
        .select(["x_pixel", "y_pixel", "integrity"])
    )
    if pixel_integrity.height == 0:
        return result

    cutoff = _sorted_value_knee(
        pixel_integrity.get_column("integrity").to_numpy().astype(np.float64, copy=False)
    )
    if not np.isfinite(cutoff):
        return result

    hotspot_pixels = (
        pixel_integrity.filter(pl.col("integrity") <= cutoff)
        .select(["x_pixel", "y_pixel"])
        .unique()
    )
    result["signal_hotspot_cutoff_fast"] = float(cutoff)
    result["signal_hotspot_pixels_used_fast"] = int(hotspot_pixels.height)
    if hotspot_pixels.height == 0:
        return result

    cell_hotspots = (
        assigned_binned.select([cell_id_column, "x_pixel", "y_pixel"])
        .unique()
        .join(hotspot_pixels, on=["x_pixel", "y_pixel"], how="inner")
        .group_by(cell_id_column)
        .agg(pl.len().alias("hotspot_pixel_count"))
    )
    if cell_hotspots.height == 0:
        return result

    if max_cells > 0 and cell_hotspots.height > max_cells:
        rng = np.random.default_rng(seed)
        ids = np.array(cell_hotspots.get_column(cell_id_column).to_list(), dtype=object)
        picked = rng.choice(ids, size=max_cells, replace=False).tolist()
        cell_hotspots = cell_hotspots.filter(pl.col(cell_id_column).is_in(picked))

    result["signal_hotspot_candidate_cells_fast"] = int(cell_hotspots.height)

    candidate_tx = assigned_binned.join(
        cell_hotspots.select([cell_id_column]),
        on=cell_id_column,
        how="inner",
    )
    if candidate_tx.height == 0:
        return result

    cell_z = (
        candidate_tx.group_by([cell_id_column, "z_index"])
        .agg(pl.len().alias("n_z"))
        .sort([cell_id_column, "z_index"])
        .with_columns(
            pl.col("n_z").sum().over(cell_id_column).alias("n_total_tx"),
            pl.col("n_z").cum_sum().over(cell_id_column).alias("cum_n"),
        )
    )
    split_z = (
        cell_z.filter(pl.col("cum_n") >= (pl.col("n_total_tx") / 2.0))
        .group_by(cell_id_column)
        .agg(pl.min("z_index").alias("split_z"))
    )
    cell_side_counts = (
        cell_z.join(split_z, on=cell_id_column, how="inner")
        .group_by(cell_id_column)
        .agg(
            (
                pl.when(pl.col("z_index") <= pl.col("split_z"))
                .then(pl.col("n_z"))
                .otherwise(0)
            )
            .sum()
            .alias("n_lower"),
            (
                pl.when(pl.col("z_index") > pl.col("split_z"))
                .then(pl.col("n_z"))
                .otherwise(0)
            )
            .sum()
            .alias("n_upper"),
            pl.first("n_total_tx").alias("n_total_tx"),
            pl.first("split_z").alias("split_z"),
        )
    )

    eligible_cells = (
        cell_hotspots.join(cell_side_counts, on=cell_id_column, how="inner")
        .filter(pl.col("n_total_tx") >= min_transcripts_per_cell)
        .with_columns(
            pl.when(pl.col("n_lower") < pl.col("n_upper"))
            .then(pl.col("n_lower"))
            .otherwise(pl.col("n_upper"))
            .alias("n_minor_side")
        )
    )
    result["signal_hotspot_metric_cells_used_fast"] = int(eligible_cells.height)
    if eligible_cells.height == 0:
        return result

    scorable_cells = eligible_cells.filter(pl.col("n_minor_side") >= min_side_transcripts)
    result["signal_hotspot_cells_scored_fast"] = int(scorable_cells.height)

    scores = eligible_cells.select([cell_id_column, "hotspot_pixel_count"]).with_columns(
        pl.lit(1.0).alias("coherence")
    )

    if scorable_cells.height > 0:
        candidate_gene_z = (
            candidate_tx.group_by([cell_id_column, "z_index", feature_column])
            .agg(pl.len().alias("count"))
            .join(scorable_cells.select([cell_id_column, "split_z"]), on=cell_id_column, how="inner")
            .with_columns(
                pl.when(pl.col("z_index") <= pl.col("split_z"))
                .then(pl.lit("lower"))
                .otherwise(pl.lit("upper"))
                .alias("side")
            )
            .group_by([cell_id_column, "side", feature_column])
            .agg(pl.sum("count").alias("count"))
        )

        lower_side = candidate_gene_z.filter(pl.col("side") == "lower").select(
            [cell_id_column, feature_column, pl.col("count").alias("count_lower")]
        )
        upper_side = candidate_gene_z.filter(pl.col("side") == "upper").select(
            [cell_id_column, feature_column, pl.col("count").alias("count_upper")]
        )
        dot = (
            lower_side.join(
                upper_side,
                on=[cell_id_column, feature_column],
                how="inner",
            )
            .group_by(cell_id_column)
            .agg(
                (pl.col("count_lower") * pl.col("count_upper"))
                .sum()
                .cast(pl.Float64)
                .alias("dot")
            )
        )
        lower_norm = (
            lower_side.group_by(cell_id_column)
            .agg(((pl.col("count_lower") * pl.col("count_lower")).sum()).alias("norm_sq_lower"))
            .with_columns(pl.col("norm_sq_lower").cast(pl.Float64))
        )
        upper_norm = (
            upper_side.group_by(cell_id_column)
            .agg(((pl.col("count_upper") * pl.col("count_upper")).sum()).alias("norm_sq_upper"))
            .with_columns(pl.col("norm_sq_upper").cast(pl.Float64))
        )
        scored = (
            scorable_cells.select([cell_id_column])
            .join(lower_norm, on=cell_id_column, how="inner")
            .join(upper_norm, on=cell_id_column, how="inner")
            .join(dot, on=cell_id_column, how="left")
            .with_columns(pl.col("dot").fill_null(0.0))
            .with_columns(
                (
                    pl.col("dot")
                    / (
                        (pl.col("norm_sq_lower").sqrt() * pl.col("norm_sq_upper").sqrt())
                        + 1e-9
                    )
                )
                .clip(lower_bound=0.0, upper_bound=1.0)
                .alias("coherence")
            )
            .select([cell_id_column, "coherence"])
        )
        scores = (
            scores.join(scored, on=cell_id_column, how="left", suffix="_new")
            .with_columns(pl.col("coherence_new").fill_null(pl.col("coherence")).alias("coherence"))
            .drop("coherence_new")
        )

    weights = scores.get_column("hotspot_pixel_count").to_numpy().astype(np.float64, copy=False)
    coherence = scores.get_column("coherence").to_numpy().astype(np.float64, copy=False)
    flags = (coherence < doublet_threshold).astype(np.float64)
    result["signal_hotspot_doublet_fraction_fast"] = float(np.average(flags, weights=weights))
    result["signal_hotspot_doublet_fraction_ci95_fast"] = float(
        _weighted_bernoulli_ci95(flags, weights)
    )
    return result


def load_me_gene_pairs(
    *,
    me_gene_pairs_path: Optional[Path] = None,
    scrna_reference_path: Optional[Path] = None,
    scrna_celltype_column: str = "cell_type",
) -> list[tuple[str, str]]:
    """Load mutually exclusive gene pairs from file or scRNA reference."""
    if me_gene_pairs_path is not None:
        pairs: list[tuple[str, str]] = []
        with Path(me_gene_pairs_path).open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    parts = [p.strip() for p in line.split("\t")]
                elif "," in line:
                    parts = [p.strip() for p in line.split(",")]
                else:
                    parts = [p.strip() for p in line.split()]
                if len(parts) < 2:
                    continue
                if parts[0].lower() in {"gene1", "gene_a"} and parts[1].lower() in {"gene2", "gene_b"}:
                    continue
                pairs.append((parts[0], parts[1]))
        return pairs

    if scrna_reference_path is not None:
        pairs, _ = load_me_genes_from_scrna(
            scrna_path=Path(scrna_reference_path),
            cell_type_column=scrna_celltype_column,
            gene_name_column="feature_name",
        )
        return [(str(g1), str(g2)) for g1, g2 in pairs]

    return []


def compute_mecr_fast(
    anndata_path: Path,
    gene_pairs: Sequence[tuple[str, str]],
    *,
    max_pairs: int = 500,
    soft: bool = True,
    seed: int = 0,
) -> dict[str, float]:
    """Compute fast MECR from an AnnData file (lower is better)."""
    empty = {"mecr_fast": float("nan"), "mecr_ci95_fast": float("nan"), "mecr_pairs_used": 0}
    if anndata_path is None or not Path(anndata_path).exists():
        return empty
    if len(gene_pairs) == 0:
        return empty

    adata = None
    try:
        adata = ad.read_h5ad(anndata_path, backed="r")
        var_names = [str(g) for g in adata.var_names]
        gene_to_idx_exact, gene_to_idx_norm = _build_gene_index_map(var_names)

        valid_pairs: list[tuple[int, int]] = []
        for g1, g2 in gene_pairs:
            g1s = str(g1)
            g2s = str(g2)
            i = gene_to_idx_exact.get(g1s)
            if i is None:
                i = gene_to_idx_norm.get(_normalize_gene_token(g1s))
            j = gene_to_idx_exact.get(g2s)
            if j is None:
                j = gene_to_idx_norm.get(_normalize_gene_token(g2s))
            if i is None or j is None or i == j:
                continue
            valid_pairs.append((int(i), int(j)))

        if len(valid_pairs) == 0:
            return empty

        if max_pairs > 0 and len(valid_pairs) > max_pairs:
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(valid_pairs), size=max_pairs, replace=False)
            valid_pairs = [valid_pairs[int(i)] for i in pick]

        # Load only the columns needed for sampled pairs to cap memory usage.
        unique_cols = sorted({idx for pair in valid_pairs for idx in pair})
        if len(unique_cols) == 0:
            return empty
        col_to_local = {col: pos for pos, col in enumerate(unique_cols)}
        local_pairs = [(col_to_local[i], col_to_local[j]) for i, j in valid_pairs]

        try:
            X = adata[:, unique_cols].X
        except Exception as exc:
            # Some backed AnnData stores fail indexed reads ("Cannot validate indices").
            # Fall back to in-memory loading before giving up.
            print(
                f"[segger][mecr] WARN backed read failed for {anndata_path}: {exc}; "
                "retrying in-memory"
            )
            try:
                if getattr(adata, "isbacked", False):
                    adata.file.close()
            except Exception:
                pass
            adata = ad.read_h5ad(anndata_path)
            X = adata[:, unique_cols].X

        if hasattr(X, "to_memory"):
            X = X.to_memory()
        is_sparse = sparse.issparse(X)
        if is_sparse:
            X = X.tocsc()
        else:
            X = np.asarray(X)

        vals: list[float] = []
        for i, j in local_pairs:
            if is_sparse:
                a = np.asarray(X.getcol(i).toarray()).ravel()
                b = np.asarray(X.getcol(j).toarray()).ravel()
            else:
                a = np.asarray(X[:, i]).ravel()
                b = np.asarray(X[:, j]).ravel()

            if soft:
                den = float(np.maximum(a, b).sum())
                if den <= 0:
                    continue
                num = float(np.minimum(a, b).sum())
                vals.append(num / den)
            else:
                a_bin = a > 0
                b_bin = b > 0
                either = float((a_bin | b_bin).sum())
                if either <= 0:
                    continue
                both = float((a_bin & b_bin).sum())
                vals.append(both / either)

        if len(vals) == 0:
            return {"mecr_fast": float("nan"), "mecr_ci95_fast": float("nan"), "mecr_pairs_used": 0}

        arr = np.asarray(vals, dtype=np.float64)
        ci95 = float("nan")
        if arr.size > 1:
            se = float(np.std(arr, ddof=1)) / np.sqrt(float(arr.size))
            ci95 = float(1.96 * se)

        return {
            "mecr_fast": float(np.mean(arr)),
            "mecr_ci95_fast": float(ci95),
            "mecr_pairs_used": int(len(vals)),
        }
    except Exception as exc:
        print(f"[segger][mecr] WARN failed for {anndata_path}: {exc}")
        return empty
    finally:
        try:
            if adata is not None and getattr(adata, "isbacked", False):
                adata.file.close()
        except Exception:
            pass
