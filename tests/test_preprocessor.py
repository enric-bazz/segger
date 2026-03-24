from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pandas as pd
import pytest


def _load_preprocessor_module():
    pytest.importorskip("geopandas")
    pytest.importorskip("polars")
    src_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src_root))
    try:
        return importlib.import_module("segger.io.preprocessor")
    finally:
        sys.path.pop(0)


def test_build_boundary_index_casts_numeric_ids_to_strings() -> None:
    preprocessor = _load_preprocessor_module()
    std = preprocessor.StandardBoundaryFields()

    boundaries = pd.DataFrame(
        {
            std.id: [101, 101],
            std.boundary_type: [std.nucleus_value, std.cell_value],
        }
    )

    index = preprocessor._build_boundary_index(boundaries)

    assert list(index) == ["101_0", "101_1"]
    assert index.dtype == object


def test_merscope_preprocessor_supports_transcripts_parquet_schema(tmp_path: Path) -> None:
    preprocessor = _load_preprocessor_module()
    pl = pytest.importorskip("polars")

    tx_file = tmp_path / "transcripts.parquet"
    pl.DataFrame(
        {
            "global_x": [0.0, 1.0, 10.0, 11.0],
            "global_y": [0.0, 1.0, 10.0, 11.0],
            "gene": ["A", "A", "B", "B"],
            "nucleus_boundaries_id": ["n1", None, "n2", None],
            "cell_boundaries_id": ["c1", "c1", "c2", "c2"],
            "score": [1.0, 0.8, 0.9, 0.7],
        }
    ).write_parquet(tx_file)

    pp = preprocessor.get_preprocessor(tmp_path)
    assert isinstance(pp, preprocessor.MerscopePreprocessor)

    tx = pp.transcripts
    std_tx = preprocessor.StandardTranscriptFields()
    assert std_tx.x in tx.columns
    assert std_tx.y in tx.columns
    assert std_tx.feature in tx.columns
    assert std_tx.cell_id in tx.columns
    assert std_tx.compartment in tx.columns
    assert tx[std_tx.cell_id].null_count() == 0

    bd = pp.boundaries
    std_bd = preprocessor.StandardBoundaryFields()
    assert std_bd.id in bd.columns
    assert std_bd.boundary_type in bd.columns
    assert std_bd.contains_nucleus in bd.columns
    assert len(bd) > 0
    assert set(bd[std_bd.boundary_type].unique()) == {
        std_bd.cell_value,
        std_bd.nucleus_value,
    }


def test_merscope_preprocessor_supports_entityid_assignments(tmp_path: Path) -> None:
    preprocessor = _load_preprocessor_module()
    pl = pytest.importorskip("polars")

    tx_file = tmp_path / "detected_transcripts.csv"
    pl.DataFrame(
        {
            "global_x": [0.0, 1.0, 10.0, 11.0],
            "global_y": [0.0, 1.0, 10.0, 11.0],
            "global_z": [0.0, 0.0, 0.0, 0.0],
            "gene": ["A", "A", "B", "B"],
            "EntityID": [101, 101, -1, 202],
            "score": [0.9, 0.8, 0.7, 0.95],
        }
    ).write_csv(tx_file)

    pp = preprocessor.get_preprocessor(tmp_path)
    assert isinstance(pp, preprocessor.MerscopePreprocessor)

    tx = pp.transcripts
    std_tx = preprocessor.StandardTranscriptFields()
    assert std_tx.cell_id in tx.columns
    assert tx[std_tx.cell_id].null_count() == 1

    assigned = set(
        tx.filter(tx[std_tx.cell_id].is_not_null())
        .get_column(std_tx.cell_id)
        .to_list()
    )
    assert assigned == {"101", "202"}

    bd = pp.boundaries
    std_bd = preprocessor.StandardBoundaryFields()
    assert set(bd[std_bd.id].astype(str).unique()) == {"101", "202"}


def test_merscope_preprocessor_handles_coordinate_alias_collisions(tmp_path: Path) -> None:
    preprocessor = _load_preprocessor_module()

    tx_file = tmp_path / "detected_transcripts.csv"
    tx_file.write_text(
        (
            "global_x,global_y,x,y,gene,EntityID,score\n"
            "0.0,1.0,100.0,101.0,A,101,0.9\n"
            "2.0,3.0,200.0,201.0,B,202,0.8\n"
        ),
        encoding="utf-8",
    )

    pp = preprocessor.get_preprocessor(tmp_path)
    assert isinstance(pp, preprocessor.MerscopePreprocessor)

    tx = pp.transcripts
    std_tx = preprocessor.StandardTranscriptFields()
    assert tx[std_tx.x].to_list() == [0.0, 2.0]
    assert tx[std_tx.y].to_list() == [1.0, 3.0]
    assert set(tx[std_tx.cell_id].drop_nulls().to_list()) == {"101", "202"}


def test_merscope_preprocessor_handles_duplicate_csv_headers(tmp_path: Path) -> None:
    preprocessor = _load_preprocessor_module()

    tx_file = tmp_path / "detected_transcripts.csv"
    tx_file.write_text(
        (
            "x,y,gene,EntityID,score,x\n"
            "0.0,1.0,A,101,0.9,100.0\n"
            "2.0,3.0,B,202,0.8,200.0\n"
        ),
        encoding="utf-8",
    )

    pp = preprocessor.get_preprocessor(tmp_path)
    assert isinstance(pp, preprocessor.MerscopePreprocessor)

    tx = pp.transcripts
    std_tx = preprocessor.StandardTranscriptFields()
    assert tx[std_tx.x].to_list() == [0.0, 2.0]
    assert tx[std_tx.y].to_list() == [1.0, 3.0]
    assert set(tx[std_tx.cell_id].drop_nulls().to_list()) == {"101", "202"}


def test_merscope_preprocessor_filters_blank_codewords(tmp_path: Path) -> None:
    preprocessor = _load_preprocessor_module()
    pl = pytest.importorskip("polars")

    tx_file = tmp_path / "detected_transcripts.csv"
    pl.DataFrame(
        {
            "global_x": [0.0, 1.0, 2.0],
            "global_y": [0.0, 1.0, 2.0],
            "gene": ["ACTB", "BLANK_001", "GAPDH"],
            "EntityID": [101, 101, 101],
            "score": [0.9, 0.9, 0.9],
        }
    ).write_csv(tx_file)

    pp = preprocessor.get_preprocessor(tmp_path, min_qv=0)
    tx = pp.transcripts
    std_tx = preprocessor.StandardTranscriptFields()

    assert tx[std_tx.feature].to_list() == ["ACTB", "GAPDH"]


def test_cosmx_preprocessor_supports_transcripts_parquet_schema(tmp_path: Path) -> None:
    preprocessor = _load_preprocessor_module()
    pl = pytest.importorskip("polars")

    (tmp_path / "CellLabels").mkdir()
    (tmp_path / "CompartmentLabels").mkdir()
    pd.DataFrame({"fov": [1], "x_global_px": [0.0], "y_global_px": [0.0]}).to_csv(
        tmp_path / "sample_fov_positions_file.csv",
        index=False,
    )

    pl.DataFrame(
        {
            "x_global_px": [52525.0, 63921.6],
            "y_global_px": [41902.4, -4254.0],
            "z": [2, 6],
            "target": ["TRAPPC8", "EEF1B2"],
            "cell": ["c_1_51_0", None],
            "CellComp": ["Cytoplasm", "None"],
        }
    ).write_parquet(tmp_path / "transcripts.parquet")

    pp = preprocessor.get_preprocessor(tmp_path)
    assert isinstance(pp, preprocessor.CosMXPreprocessor)

    tx = pp.transcripts
    std_tx = preprocessor.StandardTranscriptFields()
    assert std_tx.x in tx.columns
    assert std_tx.y in tx.columns
    assert std_tx.feature in tx.columns
    assert std_tx.cell_id in tx.columns
    assert std_tx.compartment in tx.columns
    assert tx[std_tx.cell_id].null_count() == 1


def test_cosmx_preprocessor_falls_back_to_synthetic_boundaries(tmp_path: Path) -> None:
    preprocessor = _load_preprocessor_module()
    pl = pytest.importorskip("polars")

    pl.DataFrame(
        {
            "x_global_px": [0.0, 1.0, 10.0, 11.0],
            "y_global_px": [0.0, 1.0, 10.0, 11.0],
            "target": ["A", "A", "B", "B"],
            "cell": ["c1", "c1", "c2", "c2"],
        }
    ).write_parquet(tmp_path / "transcripts.parquet")

    pp = preprocessor.get_preprocessor(tmp_path)
    assert isinstance(pp, preprocessor.CosMXPreprocessor)

    tx = pp.transcripts
    std_tx = preprocessor.StandardTranscriptFields()
    assert set([std_tx.x, std_tx.y, std_tx.feature, std_tx.cell_id]).issubset(set(tx.columns))
    assert tx[std_tx.cell_id].null_count() == 0

    bd = pp.boundaries
    std_bd = preprocessor.StandardBoundaryFields()
    assert len(bd) > 0
    assert set(bd[std_bd.boundary_type].unique()) == {
        std_bd.cell_value,
        std_bd.nucleus_value,
    }


def test_xenium_preprocessor_falls_back_to_synthetic_boundaries(tmp_path: Path) -> None:
    preprocessor = _load_preprocessor_module()
    pl = pytest.importorskip("polars")

    pl.DataFrame(
        {
            "x_location": [0.0, 1.0, 10.0, 11.0],
            "y_location": [0.0, 1.0, 10.0, 11.0],
            "feature_name": ["A", "A", "B", "B"],
            "cell_id": ["c1", "c1", "c2", "c2"],
            "overlaps_nucleus": [1, 0, 1, 0],
            "qv": [40, 40, 40, 40],
        }
    ).write_parquet(tmp_path / "transcripts.parquet")

    pp = preprocessor.get_preprocessor(tmp_path)
    assert isinstance(pp, preprocessor.XeniumPreprocessor)

    tx = pp.transcripts
    std_tx = preprocessor.StandardTranscriptFields()
    assert set([std_tx.x, std_tx.y, std_tx.feature, std_tx.cell_id]).issubset(set(tx.columns))
    assert tx[std_tx.cell_id].null_count() == 0

    bd = pp.boundaries
    std_bd = preprocessor.StandardBoundaryFields()
    assert len(bd) > 0
    assert set(bd[std_bd.boundary_type].unique()) == {
        std_bd.cell_value,
        std_bd.nucleus_value,
    }


def test_xenium_preprocessor_filters_control_codeword_globs(tmp_path: Path) -> None:
    preprocessor = _load_preprocessor_module()
    pl = pytest.importorskip("polars")

    pl.DataFrame(
        {
            "x_location": [0.0, 1.0, 2.0, 3.0],
            "y_location": [0.0, 1.0, 2.0, 3.0],
            "feature_name": [
                "ACTB",
                "BLANK_001",
                "NegControlCodeword123",
                "DeprecatedCodeword_probe",
            ],
            "cell_id": ["c1", "c1", "c1", "c1"],
            "overlaps_nucleus": [1, 0, 0, 0],
            "qv": [40, 40, 40, 40],
        }
    ).write_parquet(tmp_path / "transcripts.parquet")

    pp = preprocessor.get_preprocessor(tmp_path, min_qv=0)
    tx = pp.transcripts
    std_tx = preprocessor.StandardTranscriptFields()

    assert tx[std_tx.feature].to_list() == ["ACTB"]
