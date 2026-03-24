"""Tests for fragment-conditional SpatialData outputs."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pl = pytest.importorskip("polars")

from segger.export import spatialdata_writer as spatialdata_writer_module
from segger.io import spatialdata_loader as spatialdata_loader_module


class _FakeSpatialData:
    def __init__(self, points=None, shapes=None, tables=None):
        self.points = points or {}
        self.shapes = shapes or {}
        self.tables = tables or {}

    @classmethod
    def init_from_elements(cls, points=None, shapes=None, tables=None):
        return cls(points=points, shapes=shapes, tables=tables)


class _FakeSpatialDataStore:
    def __init__(self, points=None, shapes=None):
        self.points = points or {}
        self.shapes = shapes or {}
        self.attrs = {}


class _FakePointsModel:
    @staticmethod
    def parse(points, **kwargs):
        return {"points": points, "kwargs": kwargs}


class _FakeShapesModel:
    @staticmethod
    def parse(shapes, **kwargs):
        return {"shapes": shapes, "kwargs": kwargs}


class _FakeTableModel:
    @staticmethod
    def parse(table, **kwargs):
        return {"table": table, "kwargs": kwargs}


def _install_fake_spatial_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_spatialdata = types.ModuleType("spatialdata")
    fake_spatialdata.SpatialData = _FakeSpatialData

    fake_models = types.ModuleType("spatialdata.models")
    fake_models.PointsModel = _FakePointsModel
    fake_models.ShapesModel = _FakeShapesModel
    fake_models.TableModel = _FakeTableModel

    fake_dask = types.ModuleType("dask")
    fake_dask_dataframe = types.ModuleType("dask.dataframe")

    def _from_pandas(frame, npartitions=1):
        return frame

    fake_dask_dataframe.from_pandas = _from_pandas
    fake_dask.dataframe = fake_dask_dataframe

    monkeypatch.setitem(sys.modules, "spatialdata", fake_spatialdata)
    monkeypatch.setitem(sys.modules, "spatialdata.models", fake_models)
    monkeypatch.setitem(sys.modules, "dask", fake_dask)
    monkeypatch.setitem(sys.modules, "dask.dataframe", fake_dask_dataframe)
    monkeypatch.setattr(spatialdata_writer_module, "require_spatialdata", lambda: None)


def _make_transcripts(cell_ids: list[str]) -> pl.DataFrame:
    n = len(cell_ids)
    return pl.DataFrame(
        {
            "row_index": list(range(n)),
            "feature_name": ["G1"] * n,
            "x": [float(i) for i in range(n)],
            "y": [float(i) for i in range(n)],
            "segger_cell_id": cell_ids,
        }
    )


def _make_fake_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    points_df,
) -> spatialdata_loader_module.SpatialDataLoader:
    zarr_path = tmp_path / "sample.zarr"
    zarr_path.mkdir()

    fake_sdata = _FakeSpatialDataStore(points={"transcripts": points_df})
    fake_spatialdata = types.SimpleNamespace(read_zarr=lambda _path: fake_sdata)
    monkeypatch.setattr(
        spatialdata_loader_module,
        "require_spatialdata",
        lambda: fake_spatialdata,
    )
    return spatialdata_loader_module.SpatialDataLoader(path=zarr_path)


def test_spatialdata_writer_skips_fragment_shapes_and_table_when_none(monkeypatch):
    _install_fake_spatial_stack(monkeypatch)

    writer = spatialdata_writer_module.SpatialDataWriter(
        include_boundaries=True,
        boundary_method="input",
        include_table=True,
    )

    def fake_get_boundaries(*, transcripts, **kwargs):
        if len(transcripts) == 0:
            return None
        return {"n_transcripts": int(len(transcripts))}

    table_calls: list[int] = []

    def fake_build_table_element(*, transcripts, **kwargs):
        table_calls.append(int(transcripts.height))
        return {"n_rows": int(transcripts.height)}

    monkeypatch.setattr(writer, "_get_boundaries", fake_get_boundaries)
    monkeypatch.setattr(writer, "_build_table_element", fake_build_table_element)

    sdata = writer._create_spatialdata(
        transcripts=_make_transcripts(["cell-1", "cell-2"]),
        boundaries=None,
        x_column="x",
        y_column="y",
        z_column=None,
        cell_id_column="segger_cell_id",
        feature_column="feature_name",
    )

    assert writer.shapes_key in sdata.shapes
    assert writer.fragment_shapes_key not in sdata.shapes
    assert writer.table_key in sdata.tables
    assert writer.fragment_table_key not in sdata.tables
    assert table_calls == [2]


def test_spatialdata_writer_emits_fragment_shapes_and_table_when_present(monkeypatch):
    _install_fake_spatial_stack(monkeypatch)

    writer = spatialdata_writer_module.SpatialDataWriter(
        include_boundaries=True,
        boundary_method="input",
        include_table=True,
    )

    def fake_get_boundaries(*, transcripts, **kwargs):
        if len(transcripts) == 0:
            return None
        return {"n_transcripts": int(len(transcripts))}

    table_calls: list[int] = []

    def fake_build_table_element(*, transcripts, **kwargs):
        table_calls.append(int(transcripts.height))
        return {"n_rows": int(transcripts.height)}

    monkeypatch.setattr(writer, "_get_boundaries", fake_get_boundaries)
    monkeypatch.setattr(writer, "_build_table_element", fake_build_table_element)

    sdata = writer._create_spatialdata(
        transcripts=_make_transcripts(["cell-1", "fragment-5"]),
        boundaries=None,
        x_column="x",
        y_column="y",
        z_column=None,
        cell_id_column="segger_cell_id",
        feature_column="feature_name",
    )

    assert writer.shapes_key in sdata.shapes
    assert writer.fragment_shapes_key in sdata.shapes
    assert writer.table_key in sdata.tables
    assert writer.fragment_table_key in sdata.tables
    assert table_calls == [1, 1]


def test_spatialdata_loader_applies_xenium_codeword_and_qv_filters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")

    loader = _make_fake_loader(
        monkeypatch,
        tmp_path,
        points_df=pd.DataFrame(
            {
                "x_location": [0.0, 1.0, 2.0, 3.0],
                "y_location": [0.0, 1.0, 2.0, 3.0],
                "feature_name": [
                    "ACTB",
                    "BLANK_001",
                    "NegControlCodeword123",
                    "GAPDH",
                ],
                "qv": [30.0, 30.0, 30.0, 5.0],
                "cell_id": ["c1", "c1", "c1", "c1"],
                "overlaps_nucleus": [1, 0, 0, 0],
            }
        ),
    )

    tx = loader.transcripts(
        normalize=True,
        min_qv=20.0,
        apply_platform_filters=True,
    ).collect()

    assert loader.platform == "xenium"
    assert tx["feature_name"].to_list() == ["ACTB"]


def test_spatialdata_loader_applies_cosmx_control_filters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")

    loader = _make_fake_loader(
        monkeypatch,
        tmp_path,
        points_df=pd.DataFrame(
            {
                "x_global_px": [0.0, 1.0, 2.0, 3.0],
                "y_global_px": [0.0, 1.0, 2.0, 3.0],
                "target": [
                    "ACTB",
                    "NegativeProbe_1",
                    "SystemControl_1",
                    "NegPrb_1",
                ],
                "cell": ["c1", "c1", "c1", "c1"],
                "CellComp": ["Cytoplasm", "Cytoplasm", "Cytoplasm", "Cytoplasm"],
            }
        ),
    )

    tx = loader.transcripts(normalize=True, apply_platform_filters=True).collect()

    assert loader.platform == "cosmx"
    assert tx["feature_name"].to_list() == ["ACTB"]


def test_spatialdata_loader_applies_merscope_blank_filters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")

    loader = _make_fake_loader(
        monkeypatch,
        tmp_path,
        points_df=pd.DataFrame(
            {
                "global_x": [0.0, 1.0, 2.0],
                "global_y": [0.0, 1.0, 2.0],
                "gene": ["ACTB", "BLANK_001", "GAPDH"],
                "EntityID": [101, 101, 101],
            }
        ),
    )

    tx = loader.transcripts(normalize=True, apply_platform_filters=True).collect()

    assert loader.platform == "merscope"
    assert tx["feature_name"].to_list() == ["ACTB", "GAPDH"]
