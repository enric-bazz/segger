#!/usr/bin/env python3
from __future__ import annotations

"""Download or normalize a scRNA reference and standardize obs['cell_type'].

This helper is intentionally generic: you provide either an existing `.h5ad`
or a direct download URL (for example from HCA or CELLxGENE), and it writes a
normalized cached reference with a guaranteed `obs["cell_type"]` column.
"""

import argparse
import json
import sys
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad


CELL_TYPE_CANDIDATES = (
    "cell_type",
    "cell_type_name",
    "celltype",
    "cellType",
    "CellType",
    "cell_ontology_class",
    "cell_ontology_class_label",
    "broad_cell_type",
    "broad_celltype",
    "broad_type",
    "annotation",
    "annot",
    "cluster_annotation",
)

IMMUNE_KEYWORDS = (
    "t cell",
    "b cell",
    "nk",
    "monocyte",
    "macrophage",
    "dendritic",
    "neutrophil",
    "mast",
    "plasma",
    "lymphocyte",
)

NON_IMMUNE_KEYWORDS = (
    "epithelial",
    "endothelial",
    "fibroblast",
    "stromal",
    "hepatocyte",
    "neuron",
    "astrocyte",
    "oligodendrocyte",
    "acinar",
    "ductal",
    "tumor",
    "cancer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download or normalize a scRNA .h5ad reference and write a "
            "standardized cached copy with obs['cell_type']."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Direct URL to a .h5ad file.")
    source.add_argument("--input-h5ad", type=Path, help="Existing local .h5ad file.")
    parser.add_argument("--name", required=True, help="Output basename without extension.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("references"),
        help="Directory used for the standardized reference cache.",
    )
    parser.add_argument(
        "--source-label",
        default="custom",
        help="Source label stored in the metadata sidecar (e.g. hca, cellxgene).",
    )
    parser.add_argument(
        "--cell-type-column",
        help="Explicit source obs column to use. If omitted, common names are auto-detected.",
    )
    parser.add_argument(
        "--min-cell-types",
        type=int,
        default=5,
        help="Warn if fewer than this many unique cell types are present.",
    )
    parser.add_argument(
        "--allow-immune-only",
        action="store_true",
        help="Do not warn when all detected cell types look immune-specific.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the downloaded raw file under <out-dir>/raw/ instead of deleting it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing standardized reference.",
    )
    return parser.parse_args()


def download_to_cache(url: str, raw_dir: Path, name: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{name}.raw.h5ad"
    with tempfile.NamedTemporaryFile(delete=False, dir=raw_dir, suffix=".tmp") as tmp_handle:
        tmp_path = Path(tmp_handle.name)
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        tmp_path.replace(raw_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return raw_path


def detect_cell_type_column(adata: ad.AnnData, explicit: str | None) -> str:
    obs_columns = list(adata.obs.columns)
    if explicit:
        if explicit not in adata.obs:
            raise KeyError(
                f"Requested cell type column '{explicit}' was not found. "
                f"Available obs columns: {obs_columns}"
            )
        return explicit
    for candidate in CELL_TYPE_CANDIDATES:
        if candidate in adata.obs:
            return candidate
    raise KeyError(
        "Could not auto-detect a cell type column. "
        f"Available obs columns: {obs_columns}"
    )


def immune_only_guess(cell_types: list[str]) -> bool:
    if not cell_types:
        return False
    lowered = [item.lower() for item in cell_types]
    has_non_immune = any(any(token in item for token in NON_IMMUNE_KEYWORDS) for item in lowered)
    if has_non_immune:
        return False
    has_immune = any(any(token in item for token in IMMUNE_KEYWORDS) for item in lowered)
    return has_immune


def set_var_names_from_feature_column(
    adata: ad.AnnData,
    feature_column: str = "feature_name",
) -> bool:
    """Promote ``var[feature_column]`` into ``var_names`` when available."""
    if feature_column not in adata.var.columns:
        return False

    original = [str(value) for value in adata.var_names]
    feature_values = adata.var[feature_column].astype("string").fillna("")

    promoted: list[str] = []
    for fallback, raw in zip(original, feature_values.astype(str).tolist()):
        token = raw.strip()
        if not token or token.lower() in {"nan", "none"}:
            promoted.append(fallback)
        else:
            promoted.append(token)

    changed = promoted != original
    if changed:
        adata.var_names = promoted
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
        changed = True
    return changed


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"

    standardized_path = out_dir / f"{args.name}.h5ad"
    metadata_path = out_dir / f"{args.name}.json"
    if standardized_path.exists() and not args.force:
        print(
            f"ERROR: Refusing to overwrite existing file: {standardized_path}. "
            "Use --force to replace it.",
            file=sys.stderr,
        )
        return 2

    if args.input_h5ad:
        input_path = args.input_h5ad.resolve()
        source_url = ""
    else:
        source_url = args.url or ""
        input_path = download_to_cache(source_url, raw_dir, args.name)

    if not input_path.exists():
        print(f"ERROR: Input .h5ad not found: {input_path}", file=sys.stderr)
        return 2

    adata = ad.read_h5ad(input_path)
    selected_col = detect_cell_type_column(adata, args.cell_type_column)

    normalized = adata.obs[selected_col].astype("string").fillna("Unknown")
    adata.obs["cell_type"] = normalized.astype("category")
    set_var_names_from_feature_column(adata, feature_column="feature_name")

    cell_types = [str(item) for item in adata.obs["cell_type"].cat.categories.tolist()]
    warnings: list[str] = []
    if len(cell_types) < args.min_cell_types:
        warnings.append(
            f"Only {len(cell_types)} unique cell types detected; "
            f"expected at least {args.min_cell_types}."
        )
    immune_only = immune_only_guess(cell_types)
    if immune_only and not args.allow_immune_only:
        warnings.append(
            "All detected cell types look immune-specific. "
            "Double-check that the reference is not subset-restricted."
        )

    standardized_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(standardized_path)

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_label": args.source_label,
        "source_url": source_url,
        "input_h5ad": str(input_path),
        "standardized_h5ad": str(standardized_path),
        "selected_cell_type_column": selected_col,
        "normalized_cell_type_column": "cell_type",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_cell_types": len(cell_types),
        "cell_type_preview": cell_types[:20],
        "immune_only_guess": immune_only,
        "warnings": warnings,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if args.url and not args.keep_raw:
        input_path.unlink(missing_ok=True)

    print(f"standardized_h5ad={standardized_path}")
    print(f"metadata_json={metadata_path}")
    print(f"selected_cell_type_column={selected_col}")
    if warnings:
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
