"""Fetch and cache scRNA-seq references from CellxGENE Census.

This module provides a high-level API for downloading tissue-specific
scRNA-seq references, subsampling them, and caching locally so that
Segger's alignment loss and validation metrics can use them without
manual data wrangling.

Usage
-----
>>> from segger.data.atlas import fetch_reference
>>> ref = fetch_reference("colon")
>>> ref.h5ad_path
PosixPath('/path/to/cwd/.segger_references/homo_sapiens/large_intestine/large_intestine.h5ad')

The Census dependency (``cellxgene-census``) is optional.  Functions that
need it will raise ``ImportError`` with installation instructions if it
is missing.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    import anndata as ad

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIRNAME = ".segger_references"

_TISSUE_ALIASES: dict[str, str] = {
    # colorectal
    "crc": "large intestine",
    "colon": "large intestine",
    "colorectal": "large intestine",
    "rectum": "large intestine",
    # breast
    "breast": "breast",
    "mammary": "breast",
    # brain
    "brain": "brain",
    "cerebral cortex": "brain",
    "cerebellum": "brain",
    # lung
    "lung": "lung",
    "pulmonary": "lung",
    # liver
    "liver": "liver",
    "hepatic": "liver",
    # kidney
    "kidney": "kidney",
    "renal": "kidney",
    # pancreas
    "pancreas": "pancreas",
    "pancreatic": "pancreas",
    # skin
    "skin": "skin",
    "dermis": "skin",
    "epidermis": "skin",
    # heart
    "heart": "heart",
    "cardiac": "heart",
    # eye
    "eye": "eye",
    "retina": "eye",
    # stomach
    "stomach": "stomach",
    "gastric": "stomach",
    # prostate
    "prostate": "prostate",
    # bladder
    "bladder": "bladder",
    "urinary bladder": "bladder",
    # blood
    "blood": "blood",
    "pbmc": "blood",
    # bone marrow
    "bone marrow": "bone marrow",
    # lymph node
    "lymph node": "lymph node",
    # spleen
    "spleen": "spleen",
    # thymus
    "thymus": "thymus",
    # small intestine
    "small intestine": "small intestine",
    "ileum": "small intestine",
    "jejunum": "small intestine",
    "duodenum": "small intestine",
}

# All known input names (aliases + canonical Census tissue_general values)
_ALL_KNOWN_NAMES: list[str] = sorted(set(_TISSUE_ALIASES.keys()) | set(_TISSUE_ALIASES.values()))

_ORGANISM_ALIASES: dict[str, str] = {
    # human
    "human": "homo_sapiens",
    "h sapiens": "homo_sapiens",
    "h. sapiens": "homo_sapiens",
    "homo sapiens": "homo_sapiens",
    "homo_sapiens": "homo_sapiens",
    "homosapiens": "homo_sapiens",
    "homosapiense": "homo_sapiens",
    "hsapiens": "homo_sapiens",
    # mouse
    "mouse": "mus_musculus",
    "m musculus": "mus_musculus",
    "m. musculus": "mus_musculus",
    "mus musculus": "mus_musculus",
    "mus_musculus": "mus_musculus",
    "musmusculus": "mus_musculus",
    "mmusculus": "mus_musculus",
    # zebrafish
    "zebrafish": "danio_rerio",
    "danio rerio": "danio_rerio",
    "danio_rerio": "danio_rerio",
}
_ALL_KNOWN_ORGANISMS: list[str] = sorted(
    set(_ORGANISM_ALIASES.keys()) | set(_ORGANISM_ALIASES.values())
)

IMMUNE_KEYWORDS: tuple[str, ...] = (
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

NON_IMMUNE_KEYWORDS: tuple[str, ...] = (
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

UNKNOWN_CELL_TYPE_LABELS: frozenset[str] = frozenset(
    {"", "na", "n/a", "nan", "none", "unassigned", "unknown"}
)
OTHER_CELL_TYPE_LABELS: frozenset[str] = frozenset({"other"})

# Ordered coarse rules (first match wins).
_AUTO_COARSE_CELL_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Stem/Progenitor", ("transit amplifying", "stem", "progenitor", "precursor")),
    ("Plasma", ("plasma",)),
    ("Mast", ("mast",)),
    (
        "Myeloid",
        ("monocyte", "macrophage", "dendritic", "myeloid", "neutrophil"),
    ),
    ("Erythroid", ("erythro", "megakary")),
    ("Endothelial", ("endothelial",)),
    (
        "Stromal",
        (
            "fibroblast",
            "stromal",
            "pericyte",
            "perivascular",
            "stellate",
            "smooth muscle",
            "mesenchymal",
            "adipocyte",
            "vascular associated smooth muscle",
        ),
    ),
    ("Neural/Glial", ("neuron", "astrocyte", "oligodendro", "glial", "microglia")),
    (
        "Endocrine",
        (
            "endocrine",
            "islet",
            "pancreatic a cell",
            "type b pancreatic cell",
            "pancreatic b cell",
            "pancreatic d cell",
            "pancreatic pp cell",
            "pancreatic epsilon",
            "enteroendocrine",
        ),
    ),
    (
        "Epithelial",
        (
            "epithelial",
            "enterocyte",
            "colonocyte",
            "goblet",
            "tuft",
            "secretory",
            "absorptive",
            "acinar",
            "ductal",
            "basal",
            "luminal",
            "alveolar",
            "club",
            "ciliated",
            "lactocyte",
            "hepatocyte",
            "cholangiocyte",
        ),
    ),
    ("Immune", ("leukocyte", "lymphocyte", "immune cell")),
)

_B_CELL_PATTERN = re.compile(
    r"\b(?:b cell|germinal center b cell|memory b cell|mature b cell)\b"
)
_T_NK_ILC_PATTERN = re.compile(
    r"\b(?:"
    r"t cell|"
    r"alpha-beta t cell|"
    r"gamma-delta t cell|"
    r"natural killer|"
    r"natural killer cell|"
    r"nk cell|"
    r"nk t cell|"
    r"nkt cell|"
    r"innate lymphoid|"
    r"innate lymphoid cell"
    r")\b"
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtlasReference:
    """Metadata for a cached scRNA-seq reference."""

    h5ad_path: Path
    metadata_path: Path
    tissue: str
    organism: str
    census_version: str
    n_obs: int
    n_cell_types: int
    cell_type_column: str  # always "cell_type"
    immune_only: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_cache_dir(cache_dir: Path | None = None) -> Path:
    """Return the cache directory, respecting env-var override."""
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get("SEGGER_REFERENCE_CACHE_DIR")
    if env:
        return Path(env)
    return Path.cwd() / _DEFAULT_CACHE_DIRNAME


def _normalize_tissue(tissue: str) -> str:
    """Map user-friendly tissue names to Census ``tissue_general`` values.

    Raises ``ValueError`` with fuzzy-match suggestions if the name is not
    recognized.
    """
    key = " ".join(str(tissue).strip().lower().replace("_", " ").replace("-", " ").split())
    if key in _TISSUE_ALIASES:
        return _TISSUE_ALIASES[key]
    canonical = set(_TISSUE_ALIASES.values())
    if key in canonical:
        return key

    # Not a known alias — try fuzzy matching
    close = difflib.get_close_matches(key, _ALL_KNOWN_NAMES, n=5, cutoff=0.6)
    valid_tissues = sorted(canonical)

    msg = f"Unknown tissue type: '{tissue}'."
    if close:
        suggestions = ", ".join(f"'{s}'" for s in close)
        msg += f"\n  Did you mean: {suggestions}?"
    msg += f"\n  Valid tissue types: {', '.join(valid_tissues)}"
    msg += f"\n  Valid aliases: {', '.join(sorted(_TISSUE_ALIASES.keys()))}"
    raise ValueError(msg)


def _normalize_tissue_lenient(tissue: str) -> str:
    """Like ``_normalize_tissue`` but passes through unknown names instead of raising."""
    key = " ".join(str(tissue).strip().lower().replace("_", " ").replace("-", " ").split())
    canonical = set(_TISSUE_ALIASES.values())
    if key in canonical:
        return key
    return _TISSUE_ALIASES.get(key, key)


def _normalize_organism(organism: str) -> str:
    """Map user organism input to Census organism values with typo correction."""
    key = " ".join(str(organism).strip().lower().replace("_", " ").split())
    if key in _ORGANISM_ALIASES:
        return _ORGANISM_ALIASES[key]

    close = difflib.get_close_matches(key, _ALL_KNOWN_ORGANISMS, n=5, cutoff=0.6)
    canonical = sorted(set(_ORGANISM_ALIASES.values()))

    msg = f"Unknown organism: '{organism}'."
    if close:
        suggestions = ", ".join(f"'{s}'" for s in close)
        msg += f"\n  Did you mean: {suggestions}?"
    msg += f"\n  Valid organisms: {', '.join(canonical)}"
    msg += f"\n  Valid aliases: {', '.join(sorted(_ORGANISM_ALIASES.keys()))}"
    raise ValueError(msg)


def _immune_only_guess(cell_types: list[str]) -> bool:
    """Heuristic: True when all cell types look immune-specific."""
    if not cell_types:
        return False
    lowered = [ct.lower() for ct in cell_types]
    has_non_immune = any(
        any(tok in ct for tok in NON_IMMUNE_KEYWORDS) for ct in lowered
    )
    if has_non_immune:
        return False
    has_immune = any(
        any(tok in ct for tok in IMMUNE_KEYWORDS) for ct in lowered
    )
    return has_immune


def _require_census():
    """Import cellxgene_census or raise with install instructions."""
    from segger.utils.optional_deps import require_cellxgene_census
    return require_cellxgene_census()


def _open_census(census_module, census_version: str):
    """Open a Census handle across API variants.

    Recent cellxgene-census releases use ``open_soma``; older versions use
    ``open``. This helper supports both.
    """
    open_soma = getattr(census_module, "open_soma", None)
    if callable(open_soma):
        return open_soma(census_version=census_version)

    open_legacy = getattr(census_module, "open", None)
    if callable(open_legacy):
        return open_legacy(census_version=census_version)

    raise AttributeError(
        "cellxgene_census does not expose a supported opener "
        "(expected open_soma or open)."
    )


@contextmanager
def _progress_spinner(message: str, *, enabled: bool, interval_sec: float = 5.0):
    """Show progress for long calls.

    On TTYs, render a single in-place spinner line.
    On non-TTY outputs (logs/files), emit periodic heartbeat lines.
    """
    if not enabled:
        yield
        return

    stop_event = threading.Event()
    start = time.monotonic()
    stream = sys.stderr
    is_tty = bool(getattr(stream, "isatty", lambda: False)())
    tick_sec = 0.25 if is_tty else interval_sec
    lock = threading.Lock()

    def _write(text: str, *, in_place: bool = False, newline: bool = True) -> None:
        with lock:
            if in_place and is_tty:
                stream.write("\r\033[K" + text)
                if newline:
                    stream.write("\n")
            else:
                stream.write(text + ("\n" if newline else ""))
            stream.flush()

    def _worker():
        frames = "|/-\\"
        idx = 0
        while not stop_event.wait(tick_sec):
            elapsed = int(time.monotonic() - start)
            frame = frames[idx % len(frames)]
            idx += 1
            if is_tty:
                _write(f"[atlas] {message} {frame} ({elapsed}s)", in_place=True, newline=False)
            else:
                _write(f"[atlas] {message} {frame} ({elapsed}s)")

    if is_tty:
        _write(f"[atlas] {message} ...", in_place=True, newline=False)
    else:
        _write(f"[atlas] {message} ...")
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        yield
    except Exception:
        elapsed = int(time.monotonic() - start)
        _write(f"[atlas] {message} failed ({elapsed}s)", in_place=is_tty, newline=True)
        raise
    else:
        elapsed = int(time.monotonic() - start)
        _write(f"[atlas] {message} done ({elapsed}s)", in_place=is_tty, newline=True)
    finally:
        stop_event.set()
        thread.join(timeout=max(tick_sec, interval_sec) + 1.0)


def _get_anndata_compat(
    census_module,
    census_handle,
    *,
    organism: str,
    obs_value_filter: str | None,
    obs_coords=None,
):
    """Call census.get_anndata() across argument-name variants."""
    new_kwargs = {
        "organism": organism,
        "obs_value_filter": obs_value_filter,
        "obs_column_names": [
            "cell_type",
            "tissue_general",
            "dataset_id",
            "assay",
            "donor_id",
        ],
        "var_column_names": ["feature_name"],
    }
    old_kwargs = {
        "organism": organism,
        "obs_value_filter": obs_value_filter,
        "column_names": {
            "obs": [
                "cell_type",
                "tissue_general",
                "dataset_id",
                "assay",
                "donor_id",
            ],
            "var": ["feature_name"],
        },
    }
    if obs_coords is not None:
        new_kwargs["obs_coords"] = obs_coords
        old_kwargs["obs_coords"] = obs_coords

    try:
        return census_module.get_anndata(census_handle, **new_kwargs)
    except TypeError:
        try:
            return census_module.get_anndata(census_handle, **old_kwargs)
        except TypeError:
            # Older variants may not support obs_coords. Fall back to filter-only call.
            if obs_coords is None:
                raise
            new_no_coords = dict(new_kwargs)
            old_no_coords = dict(old_kwargs)
            new_no_coords.pop("obs_coords", None)
            old_no_coords.pop("obs_coords", None)
            try:
                return census_module.get_anndata(census_handle, **new_no_coords)
            except TypeError:
                return census_module.get_anndata(census_handle, **old_no_coords)


def _get_obs_compat(
    census_module,
    census_handle,
    *,
    organism: str,
    value_filter: str,
):
    """Call census.get_obs() when available; return None if not supported."""
    get_obs = getattr(census_module, "get_obs", None)
    if not callable(get_obs):
        return None
    call_kwargs = {
        "value_filter": value_filter,
        "column_names": ["soma_joinid", "cell_type"],
    }

    # Try known call styles across Census releases; if all fail, return None
    # so fetch_reference uses the legacy full-query fallback.
    for args, kwargs in (
        ((census_handle,), {"organism": organism, **call_kwargs}),
        ((census_handle, organism), call_kwargs),
    ):
        try:
            out = get_obs(*args, **kwargs)
            if hasattr(out, "to_pandas"):
                out = out.to_pandas()
            return out
        except TypeError:
            continue
    return None


def _tissue_dir(cache_dir: Path, organism: str, tissue: str) -> Path:
    """Return ``<cache_dir>/<organism>/<tissue>/``."""
    safe_tissue = tissue.replace(" ", "_")
    safe_organism = organism.replace(" ", "_")
    return cache_dir / safe_organism / safe_tissue


def _h5ad_path(cache_dir: Path, organism: str, tissue: str) -> Path:
    safe_tissue = tissue.replace(" ", "_")
    return _tissue_dir(cache_dir, organism, tissue) / f"{safe_tissue}.h5ad"


def _metadata_path(cache_dir: Path, organism: str, tissue: str) -> Path:
    safe_tissue = tissue.replace(" ", "_")
    return _tissue_dir(cache_dir, organism, tissue) / f"{safe_tissue}.json"


def _load_metadata(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _write_metadata(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _set_var_names_from_feature_column(
    adata: "ad.AnnData",
    feature_column: str = "feature_name",
) -> bool:
    """Promote ``adata.var[feature_column]`` into ``adata.var_names``.

    Empty/NA feature values fall back to existing var_names. Returns ``True``
    when var_names were updated.
    """
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


def _build_reference_from_metadata(meta_path: Path) -> AtlasReference:
    """Reconstruct an ``AtlasReference`` from the JSON sidecar."""
    meta = _load_metadata(meta_path)
    return AtlasReference(
        h5ad_path=Path(meta["h5ad_path"]),
        metadata_path=meta_path,
        tissue=meta["tissue"],
        organism=meta["organism"],
        census_version=meta.get("census_version", "unknown"),
        n_obs=meta["n_obs"],
        n_cell_types=meta["n_cell_types"],
        cell_type_column=meta.get("cell_type_column", "cell_type"),
        immune_only=meta.get("immune_only", False),
    )


def _sorted_cell_type_counts(cell_types) -> list[tuple[str, int]]:
    """Return cell-type counts sorted by descending abundance."""
    counts = cell_types.value_counts(dropna=False)
    return [(str(label), int(count)) for label, count in counts.items()]


def _filter_cell_types_by_min_count(
    obs_df,
    *,
    min_cells_per_type: int,
):
    """Drop cell-type groups with fewer than ``min_cells_per_type`` cells."""
    if min_cells_per_type < 1:
        raise ValueError("min_cells_per_type must be >= 1.")
    if min_cells_per_type <= 1:
        return obs_df, []

    counts = _sorted_cell_type_counts(obs_df["cell_type"])
    keep_labels = {label for label, count in counts if count >= min_cells_per_type}
    dropped = [(label, count) for label, count in counts if count < min_cells_per_type]
    if not dropped:
        return obs_df, []

    filtered = obs_df.loc[obs_df["cell_type"].isin(keep_labels)].copy()
    return filtered, dropped


def _cap_cell_type_counts(
    counts: list[tuple[str, int]],
    max_cell_types: int | None,
    *,
    include_other: bool = True,
) -> tuple[list[str], list[tuple[str, int]]]:
    """Reduce detailed cell types to at most ``max_cell_types`` categories.

    Returns ``(kept_labels, display_counts)`` where:
    - ``kept_labels`` are the detailed labels kept explicitly.
    - ``display_counts`` are counts for the selected output categories
      (including ``Other`` when ``include_other=True`` and collapsing occurs).
    """
    if max_cell_types is None:
        return [label for label, _ in counts], counts
    if max_cell_types < 1:
        raise ValueError("max_cell_types must be >= 1 when provided.")
    if len(counts) <= max_cell_types:
        return [label for label, _ in counts], counts

    if max_cell_types == 1:
        if not include_other:
            return [counts[0][0]], [counts[0]]
        total = sum(count for _, count in counts)
        return [], [("Other", int(total))]

    if not include_other:
        head = counts[:max_cell_types]
        kept = [label for label, _ in head]
        return kept, head

    head = counts[: max_cell_types - 1]
    other_count = int(sum(count for _, count in counts[max_cell_types - 1 :]))
    display_counts = list(head)
    if other_count > 0:
        display_counts.append(("Other", other_count))
    kept = [label for label, _ in head]
    return kept, display_counts


def _normalize_cell_type_string(cell_type: str) -> str:
    return " ".join(str(cell_type).strip().lower().split())


def _is_unknown_like_cell_type(cell_type: str) -> bool:
    return _normalize_cell_type_string(cell_type) in UNKNOWN_CELL_TYPE_LABELS


def _is_other_like_cell_type(cell_type: str) -> bool:
    return _normalize_cell_type_string(cell_type) in OTHER_CELL_TYPE_LABELS


def _auto_coarse_cell_type_label(cell_type: str) -> str:
    label = _normalize_cell_type_string(cell_type)
    if _is_unknown_like_cell_type(label):
        return "Unknown"
    if _is_other_like_cell_type(label):
        return "Other"
    if _B_CELL_PATTERN.search(label):
        return "B cell"
    if _T_NK_ILC_PATTERN.search(label):
        return "T/NK/ILC"
    for coarse_label, tokens in _AUTO_COARSE_CELL_TYPE_RULES:
        if any(token in label for token in tokens):
            return coarse_label
    return "Other"


def _build_auto_cell_type_mapping(cell_types) -> dict[str, str]:
    labels = (
        cell_types.astype("string")
        .fillna("Unknown")
        .astype(str)
        .unique()
        .tolist()
    )
    return {label: _auto_coarse_cell_type_label(label) for label in labels}


def _cell_type_drop_mask(cell_types, *, drop_unknown: bool, drop_other: bool):
    unknown_mask, other_mask = _cell_type_drop_breakdown(cell_types)
    mask = unknown_mask.copy()
    mask[:] = False
    if drop_unknown:
        mask = mask | unknown_mask
    if drop_other:
        mask = mask | other_mask
    return mask


def _cell_type_drop_breakdown(cell_types):
    normalized = (
        cell_types.astype("string")
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    unknown_mask = normalized.isin(UNKNOWN_CELL_TYPE_LABELS)
    other_mask = normalized.isin(OTHER_CELL_TYPE_LABELS)
    return unknown_mask, other_mask


def _load_cell_type_mapping(map_path: Path) -> dict[str, str]:
    """Load a fine->coarse cell-type mapping from CSV/TSV.

    Preferred column names are:
    - fine: ``cell_type``, ``cell_type_fine``, ``fine``, ``source``, ``from``
    - coarse: ``cell_type_coarse``, ``coarse_cell_type``, ``coarse``, ``target``, ``to``

    If no known column names are present, the first two columns are used.
    Empty coarse labels default to ``"Other"``.
    """
    import pandas as pd

    map_path = Path(map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"Cell-type mapping file not found: {map_path}")

    read_kwargs: dict[str, str] = {}
    if map_path.suffix.lower() in {".tsv", ".txt"}:
        read_kwargs["sep"] = "\t"

    df = pd.read_csv(map_path, **read_kwargs)
    if df.shape[1] < 2:
        raise ValueError(
            f"Cell-type mapping file must have at least 2 columns: {map_path}"
        )

    name_lookup = {str(col).strip().lower(): col for col in df.columns}
    fine_candidates = ("cell_type", "cell_type_fine", "fine", "source", "from")
    coarse_candidates = (
        "cell_type_coarse",
        "coarse_cell_type",
        "coarse",
        "target",
        "to",
    )
    fine_col = next((name_lookup[c] for c in fine_candidates if c in name_lookup), None)
    coarse_col = next((name_lookup[c] for c in coarse_candidates if c in name_lookup), None)

    if fine_col is None or coarse_col is None:
        fine_col, coarse_col = df.columns[0], df.columns[1]

    mapping: dict[str, str] = {}
    for fine, coarse in zip(df[fine_col], df[coarse_col]):
        if pd.isna(fine):
            continue
        fine_label = str(fine).strip()
        if not fine_label:
            continue
        coarse_label = "Other" if pd.isna(coarse) else str(coarse).strip()
        mapping[fine_label] = coarse_label or "Other"

    if not mapping:
        raise ValueError(
            f"Cell-type mapping file produced an empty mapping: {map_path}"
        )

    return mapping


def _apply_cell_type_mapping(
    cell_types,
    mapping: dict[str, str],
    *,
    unmapped_label: str = "Other",
):
    """Map detailed cell-type labels to coarse labels."""
    mapped = cell_types.astype("string").fillna("Unknown").map(mapping)
    return mapped.fillna(unmapped_label).astype("string")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_reference(
    tissue: str,
    *,
    cache_dir: Path | None = None,
    organism: str = "homo_sapiens",
    census_version: str = "stable",
    max_cells_per_type: int = 1000,
    min_cells_per_type: int = 1,
    max_cell_types: int | None = None,
    coarse_cell_types: bool = False,
    cell_type_map_path: Path | None = None,
    exclude_unknown: bool = False,
    drop_unknown_cell_types: bool = False,
    drop_other_cell_types: bool = False,
    min_cell_types: int = 5,
    force: bool = False,
    progress: bool = False,
) -> AtlasReference:
    """Download a tissue-specific scRNA-seq reference from CellxGENE Census.

    Parameters
    ----------
    tissue
        Tissue name (e.g. ``"colon"``, ``"breast"``, ``"brain"``).
        Common aliases are resolved automatically.
    cache_dir
        Local cache directory.  Defaults to ``./.segger_references``
        in the current working directory, or ``$SEGGER_REFERENCE_CACHE_DIR``.
    organism
        Census organism string. Friendly aliases and common typos are accepted
        (for example: ``human`` -> ``homo_sapiens``, ``mouse`` -> ``mus_musculus``).
    census_version
        Census release version (``"stable"`` recommended).
    max_cells_per_type
        Maximum cells sampled per cell type (stratified).
    min_cells_per_type
        Minimum cells required per coarse cell-type group during metadata
        prefiltering. Groups below this threshold are dropped before download.
    max_cell_types
        Optional cap on the number of output ``cell_type`` categories.
        Keeps the most abundant categories. Tail labels are collapsed into
        ``Other`` only when ``drop_other_cell_types=False``.
    coarse_cell_types
        If True, automatically map detailed labels to broad biological
        categories (e.g. ``Epithelial``, ``Myeloid``, ``Stromal``).
    cell_type_map_path
        Optional CSV/TSV mapping file for biologically meaningful coarse labels.
        If provided, Segger saves original labels as ``cell_type_fine`` and sets
        ``cell_type`` to the mapped coarse categories (unmapped labels -> ``Other``).
    exclude_unknown
        If True, exclude rows where ``cell_type == 'unknown'`` at query time.
    drop_unknown_cell_types
        If True, remove rows labeled as ``Unknown`` from final output.
    drop_other_cell_types
        If True, remove rows labeled as ``Other`` from final output.
    min_cell_types
        Emit a warning if fewer unique cell types are found.
    force
        Re-download even if a cached reference exists.
    progress
        If True, print periodic progress heartbeats during long Census calls.

    Returns
    -------
    AtlasReference
        Metadata about the cached reference, including ``h5ad_path``.

    Notes
    -----
    Metadata-first prefiltering is enabled automatically when subsetting/coarsening
    options require it (for example: ``max_cell_types``, ``min_cells_per_type > 1``,
    coarse mapping, or drop filters). In that mode, Census ``get_obs`` support is
    required so filters can be applied before matrix download.
    """
    if max_cell_types is not None and max_cell_types < 1:
        raise ValueError("max_cell_types must be >= 1 when provided.")
    if min_cells_per_type < 1:
        raise ValueError("min_cells_per_type must be >= 1.")

    resolved_cache = _get_cache_dir(cache_dir)
    normalized_organism = _normalize_organism(organism)
    normalized = _normalize_tissue(tissue)
    h5ad = _h5ad_path(resolved_cache, normalized_organism, normalized)
    meta = _metadata_path(resolved_cache, normalized_organism, normalized)
    cell_type_mapping: dict[str, str] | None = None
    resolved_map_path: Path | None = None
    mapping_mode = "none"
    include_other_bucket = not drop_other_cell_types
    effective_exclude_unknown = bool(exclude_unknown or drop_unknown_cell_types)
    requires_metadata_prefilter = bool(
        coarse_cell_types
        or (cell_type_map_path is not None)
        or (min_cells_per_type > 1)
        or (max_cell_types is not None)
        or drop_unknown_cell_types
        or drop_other_cell_types
    )
    if cell_type_map_path is not None:
        resolved_map_path = Path(cell_type_map_path).expanduser().resolve()
        cell_type_mapping = _load_cell_type_mapping(resolved_map_path)
        mapping_mode = "file"

    # Use cache if present and not forcing
    if not force and h5ad.exists() and meta.exists():
        cache_meta = _load_metadata(meta)
        cache_matches = (
            cache_meta.get("census_version", "stable") == census_version
            and cache_meta.get("max_cell_types", None) == max_cell_types
            and bool(cache_meta.get("coarse_cell_types", False)) == bool(coarse_cell_types)
            and bool(
                cache_meta.get(
                    "effective_exclude_unknown",
                    cache_meta.get("exclude_unknown", False),
                )
            ) == effective_exclude_unknown
            and bool(cache_meta.get("drop_unknown_cell_types", False)) == bool(drop_unknown_cell_types)
            and bool(cache_meta.get("drop_other_cell_types", False)) == bool(drop_other_cell_types)
            and bool(cache_meta.get("metadata_prefiltered", False)) == bool(requires_metadata_prefilter)
            and cache_meta.get("max_cells_per_type", max_cells_per_type) == max_cells_per_type
            and cache_meta.get("min_cells_per_type", min_cells_per_type) == min_cells_per_type
            and cache_meta.get("cell_type_map_path", None) == (
                str(resolved_map_path) if resolved_map_path is not None else None
            )
        )
        if cache_matches:
            if progress:
                print(f"[atlas] Using cached reference: {h5ad}", flush=True)
            return _build_reference_from_metadata(meta)
        if progress:
            print(
                "[atlas] Cache exists but fetch options changed; rebuilding reference.",
                flush=True,
            )

    census = _require_census()

    # Query Census
    import anndata as _ad
    import numpy as np

    raw_counts_preview: list[tuple[str, int]] | None = None
    selected_counts_preview: list[tuple[str, int]] | None = None
    selected_type_set: set[str] | None = None
    dropped_small_cell_types: list[tuple[str, int]] = []
    metadata_prefiltered = False

    with _progress_spinner("Opening CellxGENE Census", enabled=progress):
        census_handle = _open_census(census, census_version=census_version)

    with census_handle as c:
        # Value filter for Census query
        value_filter = (
            f"tissue_general == '{normalized}' "
            f"and is_primary_data == True "
            f"and disease == 'normal'"
        )
        if effective_exclude_unknown:
            value_filter += " and cell_type != 'unknown'"

        obs_df = None
        with _progress_spinner("Querying candidate cell metadata", enabled=progress):
            obs_df = _get_obs_compat(
                census,
                c,
                organism=normalized_organism,
                value_filter=value_filter,
            )

        if obs_df is None:
            if requires_metadata_prefilter:
                raise RuntimeError(
                    "Metadata-first atlas fetch requires cellxgene-census get_obs support. "
                    "This client version cannot prefilter by metadata before matrix download. "
                    "Upgrade cellxgene-census and retry."
                )
            with _progress_spinner("Querying and downloading reference", enabled=progress):
                adata: _ad.AnnData = _get_anndata_compat(
                    census,
                    c,
                    organism=normalized_organism,
                    obs_value_filter=value_filter,
                )
        else:
            if "soma_joinid" not in obs_df.columns:
                obs_df = obs_df.reset_index()
            if "soma_joinid" not in obs_df.columns or "cell_type" not in obs_df.columns:
                if requires_metadata_prefilter:
                    raise RuntimeError(
                        "Metadata-first atlas fetch requires get_obs columns "
                        "['soma_joinid', 'cell_type']."
                    )
                with _progress_spinner("Querying and downloading reference", enabled=progress):
                    adata: _ad.AnnData = _get_anndata_compat(
                        census,
                        c,
                        organism=normalized_organism,
                        obs_value_filter=value_filter,
                    )
            else:
                metadata_prefiltered = True
                obs_df["cell_type"] = (
                    obs_df["cell_type"].astype("string").fillna("Unknown")
                )
                if cell_type_mapping is None and coarse_cell_types:
                    cell_type_mapping = _build_auto_cell_type_mapping(obs_df["cell_type"])
                    mapping_mode = "auto"
                if cell_type_mapping is not None:
                    obs_df["cell_type_fine"] = obs_df["cell_type"]
                    obs_df["cell_type"] = _apply_cell_type_mapping(
                        obs_df["cell_type"],
                        cell_type_mapping,
                    )
                drop_mask = _cell_type_drop_mask(
                    obs_df["cell_type"],
                    drop_unknown=drop_unknown_cell_types,
                    drop_other=drop_other_cell_types,
                )
                if bool(drop_mask.any()):
                    obs_df = obs_df.loc[~drop_mask].copy()
                if len(obs_df) == 0:
                    raise ValueError(
                        f"No cells left after cell-type filtering for tissue_general='{normalized}'. "
                        f"Original query tissue: '{tissue}'. "
                        f"Try `segger atlas fetch --help` or check CellxGENE Census documentation."
                    )
                obs_df, dropped_small_cell_types = _filter_cell_types_by_min_count(
                    obs_df,
                    min_cells_per_type=min_cells_per_type,
                )
                if len(obs_df) == 0:
                    raise ValueError(
                        f"No cells left after enforcing min_cells_per_type={min_cells_per_type} "
                        f"for tissue_general='{normalized}'."
                    )
                raw_counts_preview = _sorted_cell_type_counts(obs_df["cell_type"])
                kept_labels, selected_counts_preview = _cap_cell_type_counts(
                    raw_counts_preview,
                    max_cell_types,
                    include_other=include_other_bucket,
                )
                if max_cell_types is not None and len(raw_counts_preview) > max_cell_types:
                    selected_type_set = set(kept_labels)
                    if include_other_bucket:
                        if selected_type_set:
                            obs_df["cell_type"] = obs_df["cell_type"].where(
                                obs_df["cell_type"].isin(selected_type_set),
                                other="Other",
                            )
                        else:
                            obs_df["cell_type"] = "Other"
                    else:
                        obs_df = obs_df.loc[
                            obs_df["cell_type"].isin(selected_type_set)
                        ].copy()
                rng = np.random.default_rng(42)
                sampled_joinids: list[int] = []
                for _, group in obs_df.groupby("cell_type", dropna=False):
                    ids = group["soma_joinid"].astype("int64").to_numpy()
                    if len(ids) <= max_cells_per_type:
                        sampled_joinids.extend(ids.tolist())
                    else:
                        sampled_joinids.extend(
                            rng.choice(ids, size=max_cells_per_type, replace=False).tolist()
                        )
                sampled_joinids.sort()

                with _progress_spinner("Downloading sampled reference matrix", enabled=progress):
                    adata: _ad.AnnData = _get_anndata_compat(
                        census,
                        c,
                        organism=normalized_organism,
                        obs_value_filter=None,
                        obs_coords=sampled_joinids,
                    )

    if adata.n_obs == 0:
        raise ValueError(
            f"No cells found in CellxGENE Census for tissue_general='{normalized}'. "
            f"Original query tissue: '{tissue}'. "
            f"Try `segger atlas fetch --help` or check CellxGENE Census documentation."
        )

    # Standardize cell_type column
    n_obs_downloaded = int(adata.n_obs)
    dropped_unknown_cells = 0
    dropped_other_cells = 0
    adata.obs["cell_type"] = (
        adata.obs["cell_type"].astype("string").fillna("Unknown")
    )
    if cell_type_mapping is None and coarse_cell_types:
        cell_type_mapping = _build_auto_cell_type_mapping(adata.obs["cell_type"])
        mapping_mode = "auto"
    if cell_type_mapping is not None:
        adata.obs["cell_type_fine"] = adata.obs["cell_type"]
        adata.obs["cell_type"] = _apply_cell_type_mapping(
            adata.obs["cell_type"],
            cell_type_mapping,
        )
    unknown_mask, other_mask = _cell_type_drop_breakdown(adata.obs["cell_type"])
    adata_drop_mask = (unknown_mask if drop_unknown_cell_types else (unknown_mask & False)) | (
        other_mask if drop_other_cell_types else (other_mask & False)
    )
    dropped_unknown_cells += int(unknown_mask.sum()) if drop_unknown_cell_types else 0
    dropped_other_cells += int(other_mask.sum()) if drop_other_cell_types else 0
    if bool(adata_drop_mask.any()):
        adata = adata[~adata_drop_mask.to_numpy()].copy()
    if adata.n_obs == 0:
        raise ValueError(
            "All cells were removed by cell-type filters "
            f"(drop_unknown_cell_types={drop_unknown_cell_types}, "
            f"drop_other_cell_types={drop_other_cell_types})."
        )

    # If preselection happened on metadata, enforce the same collapsing here.
    if selected_type_set is not None:
        if include_other_bucket:
            if selected_type_set:
                adata.obs["cell_type"] = adata.obs["cell_type"].where(
                    adata.obs["cell_type"].isin(selected_type_set),
                    other="Other",
                )
            else:
                adata.obs["cell_type"] = "Other"
        else:
            adata = adata[adata.obs["cell_type"].isin(selected_type_set)].copy()
    elif max_cell_types is not None:
        raw_counts_preview = _sorted_cell_type_counts(adata.obs["cell_type"])
        kept_labels, selected_counts_preview = _cap_cell_type_counts(
            raw_counts_preview,
            max_cell_types,
            include_other=include_other_bucket,
        )
        if len(raw_counts_preview) > max_cell_types:
            selected_type_set = set(kept_labels)
            if include_other_bucket:
                if selected_type_set:
                    adata.obs["cell_type"] = adata.obs["cell_type"].where(
                        adata.obs["cell_type"].isin(selected_type_set),
                        other="Other",
                    )
                else:
                    adata.obs["cell_type"] = "Other"
            else:
                adata = adata[adata.obs["cell_type"].isin(selected_type_set)].copy()

    # Re-apply Unknown/Other dropping after any max-cell-type collapsing.
    unknown_mask, other_mask = _cell_type_drop_breakdown(adata.obs["cell_type"])
    adata_drop_mask = (unknown_mask if drop_unknown_cell_types else (unknown_mask & False)) | (
        other_mask if drop_other_cell_types else (other_mask & False)
    )
    dropped_unknown_cells += int(unknown_mask.sum()) if drop_unknown_cell_types else 0
    dropped_other_cells += int(other_mask.sum()) if drop_other_cell_types else 0
    if bool(adata_drop_mask.any()):
        adata = adata[~adata_drop_mask.to_numpy()].copy()
    if adata.n_obs == 0:
        raise ValueError("All cells were removed after max_cell_types and drop filters.")

    kept_fraction = (
        float(adata.n_obs) / float(n_obs_downloaded)
        if n_obs_downloaded > 0
        else 0.0
    )
    if drop_other_cell_types and kept_fraction < 0.5:
        import warnings
        warnings.warn(
            "More than 50% of cells were dropped after coarse mapping/filtering. "
            "Consider reviewing the coarse mapping mode or disabling "
            "drop_other_cell_types for this tissue.",
            UserWarning,
            stacklevel=2,
        )

    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    cell_types = sorted(adata.obs["cell_type"].cat.categories.tolist())

    if len(cell_types) < min_cell_types:
        import warnings
        warnings.warn(
            f"Only {len(cell_types)} unique cell types for tissue '{normalized}'; "
            f"expected at least {min_cell_types}.",
            UserWarning,
            stacklevel=2,
        )

    # Safety cap for legacy fallback path that may have fetched all cells.
    if adata.n_obs > (max_cells_per_type * max(len(cell_types), 1)):
        rng = np.random.default_rng(42)
        keep_indices: list[int] = []
        for ct in cell_types:
            mask = adata.obs["cell_type"] == ct
            indices = np.where(mask)[0]
            if len(indices) <= max_cells_per_type:
                keep_indices.extend(indices.tolist())
            else:
                keep_indices.extend(
                    rng.choice(indices, size=max_cells_per_type, replace=False).tolist()
                )
        keep_indices.sort()
        adata = adata[keep_indices].copy()

    # Re-categorize after subsample
    adata.obs["cell_type"] = adata.obs["cell_type"].cat.remove_unused_categories()
    final_types = sorted(adata.obs["cell_type"].cat.categories.tolist())
    immune_only = _immune_only_guess(final_types)
    _set_var_names_from_feature_column(adata, feature_column="feature_name")

    # Atomic write
    h5ad.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False, dir=h5ad.parent, suffix=".tmp.h5ad"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        adata.write_h5ad(tmp_path)
        tmp_path.replace(h5ad)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    # Write metadata sidecar
    metadata = {
        "h5ad_path": str(h5ad),
        "tissue": normalized,
        "organism": normalized_organism,
        "census_version": census_version,
        "n_obs": int(adata.n_obs),
        "n_cell_types": len(final_types),
        "cell_type_column": "cell_type",
        "var_name_column": "feature_name",
        "immune_only": immune_only,
        "cell_type_preview": final_types[:30],
        "max_cells_per_type": max_cells_per_type,
        "min_cells_per_type": min_cells_per_type,
        "max_cell_types": max_cell_types,
        "coarse_cell_types": bool(coarse_cell_types),
        "exclude_unknown": bool(exclude_unknown),
        "effective_exclude_unknown": bool(effective_exclude_unknown),
        "drop_unknown_cell_types": bool(drop_unknown_cell_types),
        "drop_other_cell_types": bool(drop_other_cell_types),
        "metadata_prefiltered": bool(metadata_prefiltered),
        "n_obs_downloaded": int(n_obs_downloaded),
        "dropped_unknown_cells": int(dropped_unknown_cells),
        "dropped_other_cells": int(dropped_other_cells),
        "kept_fraction": float(kept_fraction),
        "coarse_mapping_mode": mapping_mode,
        "cell_type_map_path": (
            str(resolved_map_path) if resolved_map_path is not None else None
        ),
        "uses_cell_type_mapping": bool(cell_type_mapping is not None),
        "raw_n_cell_types": (
            len(raw_counts_preview) if raw_counts_preview is not None else len(final_types)
        ),
        "selected_cell_type_preview": (
            [name for name, _ in selected_counts_preview[:30]]
            if selected_counts_preview is not None
            else final_types[:30]
        ),
        "dropped_small_cell_types_preview": (
            [name for name, _ in dropped_small_cell_types[:30]]
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_metadata(meta, metadata)

    return AtlasReference(
        h5ad_path=h5ad,
        metadata_path=meta,
        tissue=normalized,
        organism=normalized_organism,
        census_version=census_version,
        n_obs=int(adata.n_obs),
        n_cell_types=len(final_types),
        cell_type_column="cell_type",
        immune_only=immune_only,
    )


def preview_reference_cell_types(
    tissue: str,
    *,
    organism: str = "homo_sapiens",
    census_version: str = "stable",
    min_cells_per_type: int = 1,
    max_cell_types: int | None = None,
    coarse_cell_types: bool = False,
    cell_type_map_path: Path | None = None,
    exclude_unknown: bool = False,
    drop_unknown_cell_types: bool = False,
    drop_other_cell_types: bool = False,
    top_n: int = 30,
    progress: bool = False,
) -> dict:
    """Preview candidate tissue cell types from Census without downloading matrix."""
    if max_cell_types is not None and max_cell_types < 1:
        raise ValueError("max_cell_types must be >= 1 when provided.")
    if top_n < 1:
        raise ValueError("top_n must be >= 1.")
    if min_cells_per_type < 1:
        raise ValueError("min_cells_per_type must be >= 1.")

    normalized = _normalize_tissue(tissue)
    normalized_organism = _normalize_organism(organism)
    census = _require_census()
    cell_type_mapping: dict[str, str] | None = None
    resolved_map_path: Path | None = None
    mapping_mode = "none"
    include_other_bucket = not drop_other_cell_types
    effective_exclude_unknown = bool(exclude_unknown or drop_unknown_cell_types)
    if cell_type_map_path is not None:
        resolved_map_path = Path(cell_type_map_path).expanduser().resolve()
        cell_type_mapping = _load_cell_type_mapping(resolved_map_path)
        mapping_mode = "file"

    with _progress_spinner("Opening CellxGENE Census", enabled=progress):
        census_handle = _open_census(census, census_version=census_version)

    with census_handle as c:
        value_filter = (
            f"tissue_general == '{normalized}' "
            f"and is_primary_data == True "
            f"and disease == 'normal'"
        )
        if effective_exclude_unknown:
            value_filter += " and cell_type != 'unknown'"
        with _progress_spinner("Querying candidate cell metadata", enabled=progress):
            obs_df = _get_obs_compat(
                census,
                c,
                organism=normalized_organism,
                value_filter=value_filter,
            )

    if obs_df is None or "cell_type" not in obs_df.columns:
        raise RuntimeError(
            "Could not preview cell types quickly because this cellxgene-census "
            "version does not expose get_obs with cell_type metadata."
        )

    obs_df["cell_type"] = obs_df["cell_type"].astype("string").fillna("Unknown")
    if cell_type_mapping is None and coarse_cell_types:
        cell_type_mapping = _build_auto_cell_type_mapping(obs_df["cell_type"])
        mapping_mode = "auto"
    if cell_type_mapping is not None:
        obs_df["cell_type"] = _apply_cell_type_mapping(
            obs_df["cell_type"],
            cell_type_mapping,
        )
    n_cells_before_drop = int(len(obs_df))
    unknown_mask, other_mask = _cell_type_drop_breakdown(obs_df["cell_type"])
    dropped_unknown_cells = int(unknown_mask.sum()) if drop_unknown_cell_types else 0
    dropped_other_cells = int(other_mask.sum()) if drop_other_cell_types else 0
    drop_mask = (unknown_mask if drop_unknown_cell_types else (unknown_mask & False)) | (
        other_mask if drop_other_cell_types else (other_mask & False)
    )
    if bool(drop_mask.any()):
        obs_df = obs_df.loc[~drop_mask].copy()
    obs_df, dropped_small_cell_types = _filter_cell_types_by_min_count(
        obs_df,
        min_cells_per_type=min_cells_per_type,
    )
    if len(obs_df) == 0:
        raise ValueError(
            "No cells left after cell-type filtering in preview mode. "
            "Adjust drop_unknown_cell_types/drop_other_cell_types/min_cells_per_type if needed."
        )
    kept_fraction = (
        float(len(obs_df)) / float(n_cells_before_drop)
        if n_cells_before_drop > 0
        else 0.0
    )
    counts = _sorted_cell_type_counts(obs_df["cell_type"])
    kept_labels, selected_counts = _cap_cell_type_counts(
        counts,
        max_cell_types,
        include_other=include_other_bucket,
    )
    selected_categories = [name for name, _ in selected_counts]

    return {
        "tissue": normalized,
        "organism": normalized_organism,
        "census_version": census_version,
        "n_cells": int(len(obs_df)),
        "n_cells_before_drop": int(n_cells_before_drop),
        "n_raw_cell_types": int(len(counts)),
        "min_cells_per_type": int(min_cells_per_type),
        "coarse_cell_types": bool(coarse_cell_types),
        "exclude_unknown": bool(exclude_unknown),
        "effective_exclude_unknown": bool(effective_exclude_unknown),
        "drop_unknown_cell_types": bool(drop_unknown_cell_types),
        "drop_other_cell_types": bool(drop_other_cell_types),
        "metadata_prefiltered": True,
        "dropped_unknown_cells": int(dropped_unknown_cells),
        "dropped_other_cells": int(dropped_other_cells),
        "kept_fraction": float(kept_fraction),
        "raw_top_cell_types": counts[:top_n],
        "max_cell_types": max_cell_types,
        "uses_cell_type_mapping": bool(cell_type_mapping is not None),
        "coarse_mapping_mode": mapping_mode,
        "cell_type_map_path": (
            str(resolved_map_path) if resolved_map_path is not None else None
        ),
        "selected_cell_type_categories": selected_categories,
        "selected_cell_type_counts": selected_counts,
        "kept_detailed_labels": kept_labels,
        "dropped_small_cell_types": dropped_small_cell_types,
    }


def resolve_reference(
    tissue_type: str | None,
    scrna_reference_path: Path | None,
    scrna_celltype_column: str = "cell_type",
    cache_dir: Path | None = None,
) -> tuple[Path | None, str]:
    """Resolve a scRNA reference from either ``--tissue-type`` or ``--scrna-reference-path``.

    Parameters
    ----------
    tissue_type
        Tissue name for Census auto-fetch.
    scrna_reference_path
        Explicit path to a local ``.h5ad`` file.
    scrna_celltype_column
        Cell type column name (used only with ``scrna_reference_path``).
    cache_dir
        Cache directory override.

    Returns
    -------
    tuple[Path | None, str]
        ``(h5ad_path, cell_type_column)`` — both ``None``/default when
        neither argument is provided.

    Raises
    ------
    ValueError
        If both ``tissue_type`` and ``scrna_reference_path`` are provided.
    """
    if tissue_type and scrna_reference_path:
        raise ValueError(
            "Cannot specify both --tissue-type and --scrna-reference-path. "
            "Use one or the other."
        )
    if tissue_type:
        ref = fetch_reference(tissue_type, cache_dir=cache_dir)
        return ref.h5ad_path, ref.cell_type_column
    if scrna_reference_path:
        return Path(scrna_reference_path), scrna_celltype_column
    return None, scrna_celltype_column


def list_cached_references(cache_dir: Path | None = None) -> list[AtlasReference]:
    """List all cached scRNA-seq references.

    Returns
    -------
    list[AtlasReference]
        One entry per cached tissue reference, sorted by tissue name.
    """
    resolved = _get_cache_dir(cache_dir)
    if not resolved.exists():
        return []

    refs: list[AtlasReference] = []
    for meta_path in sorted(resolved.rglob("*.json")):
        try:
            refs.append(_build_reference_from_metadata(meta_path))
        except (KeyError, json.JSONDecodeError):
            continue
    return refs


def clear_cache(
    tissue: str | None = None,
    cache_dir: Path | None = None,
) -> int:
    """Remove cached references.

    Parameters
    ----------
    tissue
        If provided, remove only this tissue's cache.  If ``None``,
        remove all cached references.
    cache_dir
        Cache directory override.

    Returns
    -------
    int
        Number of reference directories removed.
    """
    resolved = _get_cache_dir(cache_dir)
    if not resolved.exists():
        return 0

    if tissue is not None:
        # Use lenient normalization for cache ops (don't error on unknown names)
        normalized = _normalize_tissue_lenient(tissue)
        removed = 0
        # Search across all organism dirs
        for org_dir in resolved.iterdir():
            if not org_dir.is_dir():
                continue
            safe_tissue = normalized.replace(" ", "_")
            tissue_dir = org_dir / safe_tissue
            if tissue_dir.exists():
                shutil.rmtree(tissue_dir)
                removed += 1
        return removed

    # Clear everything
    count = 0
    for org_dir in resolved.iterdir():
        if not org_dir.is_dir():
            continue
        for tissue_dir in org_dir.iterdir():
            if tissue_dir.is_dir():
                shutil.rmtree(tissue_dir)
                count += 1
    return count
