#!/usr/bin/env python3
"""Quick terminal report of transcript counts for benchmark datasets.

It discovers datasets from benchmark roots (../benchmark_segger_lsf* and local
archive), then resolves transcript counts from segbench's dataset_overview.html
when possible, falling back to the local hardcoded map.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from plot_lsf_segger_mode_resources import (
    DATASET_LABELS,
    FALLBACK_DATASET_TRANSCRIPTS,
    discover_default_roots,
    discover_segbench_overview_path,
    load_transcript_counts_from_segbench_overview,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Print dataset transcript sizes discovered from benchmark roots."
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=None,
        help="Optional benchmark roots (each must contain datasets/).",
    )
    parser.add_argument(
        "--include-known",
        action="store_true",
        help="Also include known datasets from fallback map even if absent in roots.",
    )
    parser.add_argument(
        "--show-roots",
        action="store_true",
        help="Print semicolon-separated source roots per dataset.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Project root used for default discovery.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()

    roots = (
        args.roots
        if args.roots is not None and len(args.roots) > 0
        else discover_default_roots(project_root)
    )
    roots = [root.resolve() for root in roots if root.exists()]
    if not roots:
        raise SystemExit("No benchmark roots found. Pass explicit paths via --roots.")

    datasets_in_roots: dict[str, set[str]] = defaultdict(set)
    for root in roots:
        datasets_dir = root / "datasets"
        if not datasets_dir.is_dir():
            continue
        for dataset_dir in sorted(datasets_dir.iterdir()):
            if dataset_dir.is_dir():
                datasets_in_roots[dataset_dir.name].add(str(root))

    overview_path = discover_segbench_overview_path(project_root)
    segbench_counts: dict[str, float] = {}
    if overview_path is not None:
        segbench_counts = load_transcript_counts_from_segbench_overview(overview_path)

    datasets = set(datasets_in_roots.keys())
    if args.include_known:
        datasets.update(FALLBACK_DATASET_TRANSCRIPTS.keys())
    datasets = set(sorted(datasets))

    records: list[dict[str, str]] = []
    for dataset in datasets:
        if dataset in segbench_counts:
            transcripts = segbench_counts[dataset]
            source = "segbench_dataset_overview"
        elif dataset in FALLBACK_DATASET_TRANSCRIPTS:
            transcripts = FALLBACK_DATASET_TRANSCRIPTS[dataset]
            source = "fallback_map"
        else:
            transcripts = None
            source = "unknown"

        records.append(
            {
                "dataset": dataset,
                "label": DATASET_LABELS.get(dataset, dataset),
                "transcripts": (
                    f"{int(round(transcripts))}" if transcripts is not None else "NA"
                ),
                "transcripts_million": (
                    f"{transcripts / 1e6:.2f}" if transcripts is not None else "NA"
                ),
                "source": source,
                "roots_count": str(len(datasets_in_roots.get(dataset, set()))),
                "roots": ";".join(sorted(datasets_in_roots.get(dataset, set()))),
            }
        )

    def sort_key(row: dict[str, str]) -> tuple[int, float, str]:
        if row["transcripts"] == "NA":
            return (1, float("inf"), row["dataset"])
        return (0, float(row["transcripts"]), row["dataset"])

    records.sort(key=sort_key)

    print(f"# project_root: {project_root}")
    print("# roots:")
    for root in roots:
        print(f"#   - {root}")
    print(
        "# transcript_source_file: "
        f"{overview_path if overview_path is not None else 'not found (fallback/unknown only)'}"
    )

    header = [
        "dataset",
        "label",
        "transcripts",
        "transcripts_million",
        "source",
        "roots_count",
    ]
    if args.show_roots:
        header.append("roots")
    print("\t".join(header))

    for row in records:
        values = [row[col] for col in header]
        print("\t".join(values))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
