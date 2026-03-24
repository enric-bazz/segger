from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData

from segger.validation.quick_metrics import (
    compute_center_border_ncv_fast,
    compute_positive_marker_recall_fast,
    compute_reference_morphology_match_fast,
    compute_resolvi_contamination_fast,
    compute_signal_hotspot_doublet_fast,
    compute_spurious_coexpression_fast,
)


def _write_reference_h5ad(path) -> None:
    adata = AnnData(
        X=np.asarray(
            [
                [12, 6, 0, 0],
                [10, 5, 0, 0],
                [0, 0, 11, 7],
                [0, 0, 9, 6],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}),
        var=pd.DataFrame(index=["ga", "ga2", "gb", "gb2"]),
    )
    adata.write_h5ad(path)


def _write_reference_h5ad_with_positional_vars(path) -> None:
    adata = AnnData(
        X=np.asarray(
            [
                [12, 6, 0, 0],
                [10, 5, 0, 0],
                [0, 0, 11, 7],
                [0, 0, 9, 6],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame({"cell_type_fine": ["A", "A", "B", "B"]}),
        var=pd.DataFrame(
            {"feature_name": ["ga", "ga2", "gb", "gb2"]},
            index=["0", "1", "2", "3"],
        ),
    )
    adata.write_h5ad(path)


def _make_reference_like_transcripts() -> tuple[pl.DataFrame, pl.DataFrame]:
    rows: list[dict[str, object]] = []
    cell_specs = [
        ("cell_a1", 0.0, 0.0, "ga", "ga2"),
        ("cell_a2", 10.0, 0.0, "ga", "ga2"),
        ("cell_b1", 0.0, 100.0, "gb", "gb2"),
        ("cell_b2", 10.0, 100.0, "gb", "gb2"),
    ]
    offsets = [
        (0.0, 0.0, 1, "ga"),
        (0.0, 2.0, 1, "ga2"),
        (0.0, 4.0, 1, "hx1"),
        (2.0, 0.0, 1, "hx2"),
        (2.0, 4.0, 1, "hx3"),
        (4.0, 0.0, 1, "ga"),
        (4.0, 2.0, 1, "ga2"),
        (4.0, 4.0, 1, "hx1"),
        (1.5, 1.5, 2, "ga2"),
        (1.5, 2.5, 2, "hx2"),
        (2.5, 1.5, 2, "hx3"),
        (2.5, 2.5, 2, "ga"),
    ]

    for cell_id, base_x, base_y, major_gene, minor_gene in cell_specs:
        gene_map = {
            "ga": major_gene,
            "ga2": minor_gene,
            "hx1": "hx1",
            "hx2": "hx2",
            "hx3": "hx3",
        }
        for dx, dy, compartment, gene_key in offsets:
            rows.append(
                {
                    "cell_id": cell_id,
                    "feature_name": gene_map[gene_key],
                    "x": base_x + dx,
                    "y": base_y + dy,
                    "cell_compartment": compartment,
                }
            )

    source_tx = pl.DataFrame(rows)
    assigned_tx = source_tx.with_columns(pl.col("cell_id").alias("segger_cell_id"))
    return source_tx, assigned_tx


def _make_source_transcripts() -> pl.DataFrame:
    rows: list[dict[str, object]] = []

    for x in range(10):
        for _ in range(5):
            rows.append(
                {
                    "feature_name": "A",
                    "x": float(x),
                    "y": 0.0,
                    "z": 0.0,
                }
            )
            rows.append(
                {
                    "feature_name": "A",
                    "x": float(x),
                    "y": 0.0,
                    "z": 1.0,
                }
            )

    for x in range(10, 20):
        for _ in range(5):
            rows.append(
                {
                    "feature_name": "A",
                    "x": float(x),
                    "y": 0.0,
                    "z": 0.0,
                }
            )
            rows.append(
                {
                    "feature_name": "B",
                    "x": float(x),
                    "y": 0.0,
                    "z": 1.0,
                }
            )

    return pl.DataFrame(rows)


def test_signal_hotspot_doublet_returns_empty_without_z() -> None:
    source_tx = pl.DataFrame(
        {
            "feature_name": ["A", "A"],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
        }
    )
    assigned_tx = pl.DataFrame(
        {
            "segger_cell_id": ["c1", "c1"],
            "feature_name": ["A", "A"],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
        }
    )

    result = compute_signal_hotspot_doublet_fast(source_tx, assigned_tx)

    assert math.isnan(result["signal_hotspot_doublet_fraction_fast"])
    assert result["signal_hotspot_candidate_cells_fast"] == 0
    assert result["signal_hotspot_metric_cells_used_fast"] == 0
    assert result["signal_hotspot_cells_scored_fast"] == 0


def test_signal_hotspot_doublet_flags_merged_hotspot_cell() -> None:
    source_tx = _make_source_transcripts()
    assigned_tx = (
        source_tx.filter(pl.col("x") >= 10.0)
        .with_columns(pl.lit("merged").alias("segger_cell_id"))
        .select(["segger_cell_id", "feature_name", "x", "y", "z"])
    )

    result = compute_signal_hotspot_doublet_fast(
        source_tx,
        assigned_tx,
        grid_size=1.0,
        min_transcripts_per_cell=20,
        min_side_transcripts=5,
    )

    assert result["signal_hotspot_candidate_cells_fast"] == 1
    assert result["signal_hotspot_metric_cells_used_fast"] == 1
    assert result["signal_hotspot_cells_scored_fast"] == 1
    assert result["signal_hotspot_pixels_used_fast"] == 10
    assert result["signal_hotspot_doublet_fraction_fast"] > 0.9


def test_signal_hotspot_doublet_treats_one_sided_cell_as_non_doublet() -> None:
    source_tx = _make_source_transcripts()
    assigned_tx = (
        source_tx.filter((pl.col("x") >= 10.0) & (pl.col("z") == 0.0))
        .with_columns(pl.lit("split_lower").alias("segger_cell_id"))
        .select(["segger_cell_id", "feature_name", "x", "y", "z"])
    )

    result = compute_signal_hotspot_doublet_fast(
        source_tx,
        assigned_tx,
        grid_size=1.0,
        min_transcripts_per_cell=20,
        min_side_transcripts=5,
    )

    assert result["signal_hotspot_candidate_cells_fast"] == 1
    assert result["signal_hotspot_metric_cells_used_fast"] == 1
    assert result["signal_hotspot_cells_scored_fast"] == 0
    assert result["signal_hotspot_doublet_fraction_fast"] == 0.0


def test_reference_guided_fast_metrics_return_sane_scores(tmp_path) -> None:
    reference_path = tmp_path / "reference.h5ad"
    _write_reference_h5ad(reference_path)
    source_tx, assigned_tx = _make_reference_like_transcripts()

    marker = compute_positive_marker_recall_fast(
        assigned_tx,
        scrna_reference_path=reference_path,
        scrna_celltype_column="cell_type",
        min_transcripts_per_cell=10,
        max_cells=10,
        n_markers_per_type=2,
        min_specificity_ratio=1.1,
    )
    assert marker["positive_marker_cells_used_fast"] == 4
    assert marker["positive_marker_recall_fast"] >= 99.0

    resolvi = compute_resolvi_contamination_fast(
        assigned_tx,
        scrna_reference_path=reference_path,
        scrna_celltype_column="cell_type",
        min_transcripts_per_cell=10,
        max_cells=10,
        k_neighbors=1,
        max_neighbor_distance=20.0,
    )
    assert resolvi["resolvi_metric_cells_used"] == 4
    assert resolvi["resolvi_contamination_pct_fast"] <= 1.0

    center_border = compute_center_border_ncv_fast(
        assigned_tx,
        min_transcripts_per_cell=10,
        max_cells=10,
        n_neighbors=1,
    )
    assert center_border["center_border_ncv_cells_used_fast"] == 4
    assert 0.75 <= center_border["center_border_ncv_score_fast"] <= 1.0

    morphology = compute_reference_morphology_match_fast(source_tx, assigned_tx)
    assert morphology["reference_morphology_cells_used_fast"] == 4
    assert morphology["reference_morphology_match_fast"] >= 0.99


def test_reference_guided_fast_metrics_fallback_to_feature_name_and_celltype_alias(tmp_path) -> None:
    reference_path = tmp_path / "reference_positional.h5ad"
    _write_reference_h5ad_with_positional_vars(reference_path)
    source_tx, assigned_tx = _make_reference_like_transcripts()

    marker = compute_positive_marker_recall_fast(
        assigned_tx,
        scrna_reference_path=reference_path,
        scrna_celltype_column="cell_type",
        min_transcripts_per_cell=10,
        max_cells=10,
        n_markers_per_type=2,
        min_specificity_ratio=1.1,
    )
    assert marker["positive_marker_cells_used_fast"] == 4
    assert marker["positive_marker_recall_fast"] >= 99.0

    resolvi = compute_resolvi_contamination_fast(
        assigned_tx,
        scrna_reference_path=reference_path,
        scrna_celltype_column="cell_type",
        min_transcripts_per_cell=10,
        max_cells=10,
        k_neighbors=1,
        max_neighbor_distance=20.0,
    )
    assert resolvi["resolvi_metric_cells_used"] == 4
    assert resolvi["resolvi_contamination_pct_fast"] <= 1.0


def test_spurious_coexpression_fast_detects_merged_pair() -> None:
    source_rows = (
        [
            {
                "cell_id": "nuc_a",
                "feature_name": "A",
                "x": 0.0,
                "y": 0.0,
                "cell_compartment": 2,
            }
            for _ in range(6)
        ]
        + [
            {
                "cell_id": "nuc_b",
                "feature_name": "B",
                "x": 0.0,
                "y": 0.0,
                "cell_compartment": 2,
            }
            for _ in range(6)
        ]
    )
    source_tx = pl.DataFrame(source_rows)
    assigned_tx = source_tx.select(["feature_name", "x", "y"]).with_columns(
        pl.lit("merged").alias("segger_cell_id")
    )

    result = compute_spurious_coexpression_fast(
        source_tx,
        assigned_tx,
        min_transcripts_per_cell=1,
        max_cells=10,
        spatial_radius=1.0,
        max_spatial_transcripts=0,
        min_gene_count=1,
        min_support=1,
        ratio_cutoff=1.1,
        nuclear_max=0.01,
        min_spatial_score=0.01,
    )

    assert result["spurious_pairs_used_fast"] == 1
    assert result["spurious_coexpression_fast"] > 0.9
