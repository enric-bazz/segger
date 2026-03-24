#!/usr/bin/env python3
"""Quick transcript-size collector from run_lsf_segger_benchmark.sh input paths.

The script parses `dataset_input_dir()` in `run_lsf_segger_benchmark.sh`, then
tries to find a transcript table under each input path and prints counts.

Usage:
  python scripts/quick_count_transcripts_from_benchmark_inputs.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
from pathlib import Path

try:
    import polars as pl
except Exception:  # pragma: no cover - optional fallback
    pl = None

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional fallback if pyarrow is unavailable
    pq = None


CASE_PATH_RE = re.compile(
    r"""^\s*([a-zA-Z0-9_]+)\)\s+printf\s+'%s'\s+"([^"]+)"\s*;;\s*$"""
)

DIRECT_CANDIDATE_NAMES = (
    "transcripts.parquet",
    "transcript.parquet",
    "detected_transcripts.parquet",
    "transcripts.csv",
    "detected_transcripts.csv",
    "transcripts.tsv",
    "detected_transcripts.tsv",
    "transcripts.csv.gz",
    "detected_transcripts.csv.gz",
    "transcripts.tsv.gz",
    "detected_transcripts.tsv.gz",
)

TEXT_SUFFIXES = (".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz")
GENE_COLUMN_CANDIDATES = (
    "feature_name",
    "gene",
    "gene_name",
    "codeword",
    "codeword_index",
    "feature_id",
    "gene_id",
    "target",
)
INVALID_GENE_TOKENS = {
    "",
    "none",
    "nan",
    "null",
    "na",
    "n/a",
    "unassigned",
    "unknown",
    "-1",
    "-1.0",
}
PLATFORM_GENE_COL_CANDIDATES = {
    "xenium": ("feature_name", "gene", "target", "gene_name", "feature", "feature_id", "gene_id"),
    "merscope": ("gene", "feature_name", "target", "gene_name", "feature", "feature_id", "gene_id"),
    "cosmx": ("target", "feature_name", "gene", "gene_name", "feature", "feature_id", "gene_id"),
    "generic": GENE_COLUMN_CANDIDATES,
}
CONTROL_GENE_PATTERNS = {
    "xenium": [
        r"^NegControlProbe_",
        r"^antisense_",
        r"^NegControlCodeword",
        r"^BLANK_",
        r"^DeprecatedCodeword_",
        r"^UnassignedCodeword_",
    ],
    "cosmx": [
        r"^Negative",
        r"^SystemControl",
        r"^NegPrb",
    ],
}
LIKELY_NON_GENE_COLUMNS = {
    "x",
    "y",
    "z",
    "x_location",
    "y_location",
    "z_location",
    "global_x",
    "global_y",
    "global_z",
    "x_global_px",
    "y_global_px",
    "cell_id",
    "cell",
    "cellcomp",
    "compartment",
    "cell_compartment",
    "qv",
    "quality",
    "score",
    "overlaps_nucleus",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse dataset input paths from run_lsf_segger_benchmark.sh and "
            "print transcript counts."
        )
    )
    parser.add_argument(
        "--run-script",
        type=Path,
        default=Path("scripts/run_lsf_segger_benchmark.sh"),
        help="Path to run_lsf_segger_benchmark.sh",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Recursive search depth under each dataset input dir (default: 4).",
    )
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Only print dataset->input_dir mapping (skip file discovery/counting).",
    )
    parser.add_argument(
        "--with-genes",
        dest="with_genes",
        action="store_true",
        default=True,
        help="Count unique genes/features (default: on).",
    )
    parser.add_argument(
        "--no-genes",
        dest="with_genes",
        action="store_false",
        help="Disable gene counting and only report transcript totals.",
    )
    parser.add_argument(
        "--gene-column",
        type=str,
        default="",
        help="Optional explicit gene column name override.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Optional benchmark root (e.g. ../benchmark_segger_lsf8). "
            "If set, only dataset keys present under <dataset-root>/datasets are processed."
        ),
    )
    parser.add_argument(
        "--drop-non-ok",
        action="store_true",
        help="Drop rows where status is not 'ok'.",
    )
    return parser.parse_args()


def parse_dataset_input_paths(run_script: Path) -> list[tuple[str, str]]:
    if not run_script.is_file():
        raise FileNotFoundError(f"run script not found: {run_script}")

    lines = run_script.read_text(encoding="utf-8", errors="ignore").splitlines()

    in_func = False
    mappings: list[tuple[str, str]] = []
    for line in lines:
        if line.strip().startswith("dataset_input_dir()"):
            in_func = True
            continue

        if in_func and line.strip() == "}":
            break

        if not in_func:
            continue

        match = CASE_PATH_RE.match(line)
        if match:
            dataset, path = match.group(1), match.group(2)
            mappings.append((dataset, path))

    if not mappings:
        raise RuntimeError(
            "Could not parse dataset_input_dir() mappings from run script."
        )
    return mappings


def is_transcript_like_name(name: str) -> bool:
    lower = name.lower()
    if "transcript" not in lower:
        return False
    return lower.endswith((".parquet", ".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz"))


def relative_depth(root: Path, file_path: Path) -> int:
    rel = file_path.relative_to(root)
    return len(rel.parts) - 1


def find_transcript_file(input_dir: Path, max_depth: int) -> Path | None:
    for name in DIRECT_CANDIDATE_NAMES:
        candidate = input_dir / name
        if candidate.is_file():
            return candidate

    best: tuple[int, int, int, Path] | None = None
    for dirpath, _, filenames in os.walk(input_dir):
        base = Path(dirpath)
        try:
            depth = relative_depth(input_dir, base)
        except Exception:
            continue

        if depth > max_depth:
            continue

        for filename in filenames:
            if not is_transcript_like_name(filename):
                continue
            candidate = base / filename
            try:
                size = candidate.stat().st_size
            except OSError:
                size = 0
            # Prefer shallower files and larger files at same depth.
            rank = (depth, -size, len(filename), candidate)
            if best is None or rank < best:
                best = rank

    return best[3] if best is not None else None


def has_header(first_line: str, delimiter: str) -> bool:
    cells = [cell.strip() for cell in first_line.rstrip("\n").split(delimiter)]
    if not cells:
        return False
    numeric_like = 0
    for cell in cells:
        if cell == "":
            continue
        try:
            float(cell)
            numeric_like += 1
        except ValueError:
            pass
    # Treat as header if most cells are non-numeric.
    return numeric_like < max(1, len(cells) // 2)


def count_text_rows(path: Path) -> int:
    delimiter = "\t" if path.name.lower().endswith((".tsv", ".tsv.gz")) else ","
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    line_count = 0
    first_line = None
    with opener(path, "rt", encoding="utf-8", errors="ignore", newline="") as handle:
        for idx, line in enumerate(handle):
            if idx == 0:
                first_line = line
            line_count += 1
    if line_count == 0:
        return 0
    if first_line is not None and has_header(first_line, delimiter):
        return max(0, line_count - 1)
    return line_count


def _dataset_platform(dataset: str) -> str:
    ds = dataset.lower()
    if ds.startswith("xenium_"):
        return "xenium"
    if ds.startswith("merscope_"):
        return "merscope"
    if ds.startswith("cosmx_"):
        return "cosmx"
    return "generic"


def _pick_gene_column(
    candidates: list[str],
    *,
    dataset: str,
    explicit: str = "",
) -> str | None:
    lowered = {name.lower(): name for name in candidates}
    if explicit:
        return lowered.get(explicit.lower())
    for preferred in PLATFORM_GENE_COL_CANDIDATES.get(
        _dataset_platform(dataset), GENE_COLUMN_CANDIDATES
    ):
        if preferred in lowered:
            return lowered[preferred]
    # Fallback: pick a likely gene-like column name that is not obviously metadata.
    gene_like_priority = (
        "feature",
        "gene",
        "target",
        "codeword",
    )
    for key in gene_like_priority:
        for lower_name, original in lowered.items():
            if key in lower_name and lower_name not in LIKELY_NON_GENE_COLUMNS:
                return original
    return None


def _is_invalid_gene_token(value: str) -> bool:
    token = value.strip()
    if not token:
        return True
    lower = token.lower()
    if lower in INVALID_GENE_TOKENS:
        return True
    try:
        numeric = float(token)
    except ValueError:
        return False
    return numeric < 0


def _is_control_gene(value: str, *, dataset: str) -> bool:
    patterns = CONTROL_GENE_PATTERNS.get(_dataset_platform(dataset), [])
    for pattern in patterns:
        if re.search(pattern, value, flags=re.IGNORECASE):
            return True
    return False


def _normalize_gene_value(value: object, *, dataset: str) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if _is_invalid_gene_token(token):
        return None
    if _is_control_gene(token, dataset=dataset):
        return None
    return token


def _collect_lazy(lf: "pl.LazyFrame") -> "pl.DataFrame":
    """Collect lazy queries with streaming when available across Polars versions."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        try:
            return lf.collect(streaming=True)
        except TypeError:
            return lf.collect()


def _lazy_columns(lf: "pl.LazyFrame") -> list[str]:
    try:
        return lf.collect_schema().names()
    except AttributeError:
        return lf.columns


def _scan_with_polars(path: Path) -> "pl.LazyFrame":
    lower = path.name.lower()
    if lower.endswith(".parquet"):
        return pl.scan_parquet(path, parallel="row_groups")
    if lower.endswith((".tsv", ".tsv.gz")):
        return pl.scan_csv(path, separator="\t")
    if lower.endswith((".csv", ".csv.gz", ".txt", ".txt.gz")):
        return pl.scan_csv(path)
    raise RuntimeError(f"Unsupported file format for Polars scan: {path}")


def _normalized_gene_expr_polars(col_name: str, *, dataset: str) -> "pl.Expr":
    raw = pl.col(col_name).cast(pl.Utf8, strict=False).str.strip_chars()
    lower = raw.str.to_lowercase()
    invalid_token_expr = lower.is_in(sorted(INVALID_GENE_TOKENS))
    numeric = raw.cast(pl.Float64, strict=False)
    negative_expr = numeric.is_not_null() & (numeric < 0)
    control_expr = pl.lit(False)
    for pattern in CONTROL_GENE_PATTERNS.get(_dataset_platform(dataset), []):
        control_expr = control_expr | raw.str.contains(pattern)

    is_invalid = (
        raw.is_null()
        | raw.eq("").fill_null(False)
        | invalid_token_expr.fill_null(False)
        | negative_expr.fill_null(False)
        | control_expr.fill_null(False)
    )
    return pl.when(is_invalid).then(pl.lit(None)).otherwise(raw)


def count_with_polars(
    path: Path,
    *,
    dataset: str,
    with_genes: bool,
    gene_column: str = "",
) -> tuple[int, int | None, int | None, int | None, str | None]:
    if pl is None:
        raise RuntimeError("polars is not available")

    lf = _scan_with_polars(path)
    cols = _lazy_columns(lf)
    picked_gene_col = _pick_gene_column(cols, dataset=dataset, explicit=gene_column)

    if not with_genes or picked_gene_col is None:
        out = _collect_lazy(lf.select(pl.len().alias("n_rows")))
        n_rows = int(out.get_column("n_rows")[0])
        return n_rows, None, None, None, picked_gene_col

    raw_expr = pl.col(picked_gene_col).cast(pl.Utf8, strict=False).str.strip_chars().alias("__gene_raw")
    norm_expr = _normalized_gene_expr_polars(picked_gene_col, dataset=dataset).alias("__gene_norm")
    agg = (
        lf
        .select([raw_expr, norm_expr])
        .select(
            [
                pl.len().alias("n_rows"),
                pl.col("__gene_raw")
                .filter(pl.col("__gene_raw").is_not_null() & pl.col("__gene_raw").ne(""))
                .n_unique()
                .alias("n_genes_raw"),
                pl.col("__gene_norm").drop_nulls().n_unique().alias("n_genes"),
                pl.col("__gene_norm").is_not_null().sum().alias("n_rows_filtered"),
            ]
        )
    )
    out = _collect_lazy(agg)
    n_rows = int(out.get_column("n_rows")[0])
    n_genes = int(out.get_column("n_genes")[0])
    n_genes_raw = int(out.get_column("n_genes_raw")[0])
    n_rows_filtered = int(out.get_column("n_rows_filtered")[0])
    return n_rows, n_rows_filtered, n_genes, n_genes_raw, picked_gene_col


def count_text_rows_and_genes(
    path: Path,
    *,
    dataset: str,
    with_genes: bool,
    gene_column: str = "",
) -> tuple[int, int | None, int | None, int | None, str | None]:
    delimiter = "\t" if path.name.lower().endswith((".tsv", ".tsv.gz")) else ","
    opener = gzip.open if path.name.lower().endswith(".gz") else open

    with opener(path, "rt", encoding="utf-8", errors="ignore", newline="") as handle:
        first_line = handle.readline()
        if not first_line:
            return 0, 0 if with_genes else None, 0 if with_genes else None, 0 if with_genes else None, None

        header_present = has_header(first_line, delimiter)
        if not header_present:
            # Re-count because first line is data.
            n_rows = 1 + sum(1 for _ in handle)
            return n_rows, None, None, None, None

        header = [cell.strip() for cell in first_line.rstrip("\n").split(delimiter)]
        picked_gene_col = _pick_gene_column(
            header,
            dataset=dataset,
            explicit=gene_column,
        )
        gene_idx = header.index(picked_gene_col) if picked_gene_col in header else None
        gene_set_raw: set[str] | None = set() if (with_genes and gene_idx is not None) else None
        gene_set_valid: set[str] | None = set() if (with_genes and gene_idx is not None) else None
        n_rows_filtered = 0 if (with_genes and gene_idx is not None) else None

        n_rows = 0
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            n_rows += 1
            if gene_set_raw is not None and gene_idx is not None and gene_idx < len(row):
                raw_value = row[gene_idx].strip()
                if raw_value:
                    gene_set_raw.add(raw_value)
                norm = _normalize_gene_value(raw_value, dataset=dataset)
                if norm is not None and gene_set_valid is not None:
                    gene_set_valid.add(norm)
                    if n_rows_filtered is not None:
                        n_rows_filtered += 1

    n_genes_valid = len(gene_set_valid) if gene_set_valid is not None else None
    n_genes_raw = len(gene_set_raw) if gene_set_raw is not None else None
    return n_rows, n_rows_filtered, n_genes_valid, n_genes_raw, picked_gene_col


def count_parquet_rows_and_genes(
    path: Path,
    *,
    dataset: str,
    with_genes: bool,
    gene_column: str = "",
) -> tuple[int, int | None, int | None, int | None, str | None]:
    if pq is None:
        raise RuntimeError("pyarrow is required to read parquet.")
    parquet = pq.ParquetFile(path)
    n_rows = int(parquet.metadata.num_rows)

    if not with_genes:
        return n_rows, None, None, None, None

    schema_cols = list(parquet.schema_arrow.names)
    picked_gene_col = _pick_gene_column(
        schema_cols,
        dataset=dataset,
        explicit=gene_column,
    )
    if picked_gene_col is None:
        return n_rows, None, None, None, None

    gene_set_raw: set[str] = set()
    gene_set_valid: set[str] = set()
    n_rows_filtered = 0
    for batch in parquet.iter_batches(columns=[picked_gene_col], batch_size=500_000):
        arr = batch.column(0)
        for value in arr.to_pylist():
            if value is None:
                continue
            raw = str(value).strip()
            if raw:
                gene_set_raw.add(raw)
            norm = _normalize_gene_value(raw, dataset=dataset)
            if norm is not None:
                gene_set_valid.add(norm)
                n_rows_filtered += 1

    return n_rows, n_rows_filtered, len(gene_set_valid), len(gene_set_raw), picked_gene_col


def count_transcripts_and_genes(
    path: Path,
    *,
    dataset: str,
    with_genes: bool,
    gene_column: str = "",
) -> tuple[int, int | None, int | None, int | None, str | None]:
    # Fast path: Polars lazy engine with column pruning and vectorized distinct counting.
    if pl is not None:
        try:
            return count_with_polars(
                path,
                dataset=dataset,
                with_genes=with_genes,
                gene_column=gene_column,
            )
        except Exception:
            # Fall back to the conservative reader path below.
            pass

    lower = path.name.lower()
    if lower.endswith(".parquet"):
        return count_parquet_rows_and_genes(
            path,
            dataset=dataset,
            with_genes=with_genes,
            gene_column=gene_column,
        )
    if lower.endswith(TEXT_SUFFIXES):
        return count_text_rows_and_genes(
            path,
            dataset=dataset,
            with_genes=with_genes,
            gene_column=gene_column,
        )
    raise RuntimeError(f"Unsupported transcript file type: {path.name}")


def datasets_from_root(root: Path) -> set[str]:
    datasets_dir = root / "datasets"
    if not datasets_dir.is_dir():
        return set()
    return {
        entry.name
        for entry in datasets_dir.iterdir()
        if entry.is_dir()
    }


def main() -> int:
    args = parse_args()
    run_script = args.run_script.resolve()
    mappings = parse_dataset_input_paths(run_script)
    gene_column = (args.gene_column or "").strip()

    if args.dataset_root is not None:
        dataset_root = args.dataset_root.resolve()
        allowed = datasets_from_root(dataset_root)
        mappings = [(dataset, path) for dataset, path in mappings if dataset in allowed]

    print(f"# run_script: {run_script}")
    if args.dataset_root is not None:
        print(f"# dataset_root_filter: {args.dataset_root.resolve()}")
    print(
        "dataset\tinput_dir\tstatus\ttranscript_file\ttranscripts\ttranscripts_million\ttranscripts_filtered\ttranscripts_filtered_million\ttranscripts_removed\tgenes\tgenes_raw\tgenes_removed\tgene_column"
    )

    def emit(
        dataset: str,
        input_dir: Path,
        status: str,
        transcript_file: str = "NA",
        transcripts: str = "NA",
        transcripts_million: str = "NA",
        transcripts_filtered: str = "NA",
        transcripts_filtered_million: str = "NA",
        transcripts_removed: str = "NA",
        genes: str = "NA",
        genes_raw: str = "NA",
        genes_removed: str = "NA",
        gene_col: str = "NA",
    ) -> None:
        if args.drop_non_ok and status != "ok":
            return
        print(
            f"{dataset}\t{input_dir}\t{status}\t{transcript_file}\t"
            f"{transcripts}\t{transcripts_million}\t{transcripts_filtered}\t{transcripts_filtered_million}\t{transcripts_removed}\t"
            f"{genes}\t{genes_raw}\t{genes_removed}\t{gene_col}"
        )

    for dataset, raw_input_dir in mappings:
        input_dir = Path(raw_input_dir)
        if args.paths_only:
            emit(dataset, input_dir, "paths_only")
            continue

        if not input_dir.is_dir():
            emit(dataset, input_dir, "path_missing")
            continue

        transcript_file = find_transcript_file(input_dir, max_depth=max(0, args.max_depth))
        if transcript_file is None:
            emit(dataset, input_dir, "transcript_file_not_found")
            continue

        try:
            n_tx, n_tx_filtered, n_genes, n_genes_raw, picked_gene_col = count_transcripts_and_genes(
                transcript_file,
                dataset=dataset,
                with_genes=args.with_genes,
                gene_column=gene_column,
            )
            n_tx_m = f"{n_tx / 1e6:.2f}"
            tx_filtered_value = str(n_tx_filtered) if n_tx_filtered is not None else "NA"
            tx_filtered_m_value = f"{(n_tx_filtered / 1e6):.2f}" if n_tx_filtered is not None else "NA"
            tx_removed_value = str(max(0, n_tx - n_tx_filtered)) if n_tx_filtered is not None else "NA"
            genes_value = str(n_genes) if n_genes is not None else "NA"
            genes_raw_value = str(n_genes_raw) if n_genes_raw is not None else "NA"
            genes_removed_value = "NA"
            if n_genes is not None and n_genes_raw is not None:
                genes_removed_value = str(max(0, n_genes_raw - n_genes))
            gene_col_value = picked_gene_col if picked_gene_col else "NA"
            status = "ok"
            if args.with_genes and picked_gene_col is None:
                status = "ok_no_gene_column"
            emit(
                dataset,
                input_dir,
                status,
                transcript_file=str(transcript_file),
                transcripts=str(n_tx),
                transcripts_million=n_tx_m,
                transcripts_filtered=tx_filtered_value,
                transcripts_filtered_million=tx_filtered_m_value,
                transcripts_removed=tx_removed_value,
                genes=genes_value,
                genes_raw=genes_raw_value,
                genes_removed=genes_removed_value,
                gene_col=gene_col_value,
            )
        except Exception as exc:
            emit(
                dataset,
                input_dir,
                f"count_error:{type(exc).__name__}",
                transcript_file=str(transcript_file),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
