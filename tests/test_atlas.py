"""Tests for segger.data.atlas — no network calls required.

Uses importlib to load atlas.py directly, bypassing segger.data.__init__
which requires GPU packages (cudf, cupy, cuspatial).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


def _load_atlas_module() -> ModuleType:
    """Load atlas.py directly via importlib, avoiding heavy __init__ imports."""
    # Ensure segger.utils.optional_deps is available (no GPU deps)
    src_root = Path(__file__).resolve().parents[1] / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    # Pre-import optional_deps so the relative import works
    if "segger" not in sys.modules:
        # Create stub packages so relative imports resolve
        segger_pkg = importlib.import_module("segger")
    if "segger.utils" not in sys.modules:
        importlib.import_module("segger.utils")
    if "segger.utils.optional_deps" not in sys.modules:
        importlib.import_module("segger.utils.optional_deps")

    module_path = src_root / "segger" / "data" / "atlas.py"
    spec = importlib.util.spec_from_file_location(
        "segger.data.atlas",
        module_path,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["segger.data.atlas"] = module
    spec.loader.exec_module(module)
    return module


atlas = _load_atlas_module()


# ---------------------------------------------------------------------------
# Cache directory resolution
# ---------------------------------------------------------------------------

class TestCacheDirResolution:
    def test_explicit_cache_dir_wins(self, tmp_path, monkeypatch):
        explicit = tmp_path / "explicit_cache"
        monkeypatch.setenv("SEGGER_REFERENCE_CACHE_DIR", str(tmp_path / "env_cache"))
        assert atlas._get_cache_dir(explicit) == explicit

    def test_env_var_used_when_no_explicit_cache_dir(self, tmp_path, monkeypatch):
        env_cache = tmp_path / "env_cache"
        monkeypatch.setenv("SEGGER_REFERENCE_CACHE_DIR", str(env_cache))
        assert atlas._get_cache_dir() == env_cache

    def test_default_uses_cwd_segger_references(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SEGGER_REFERENCE_CACHE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        assert atlas._get_cache_dir() == (tmp_path / ".segger_references")


# ---------------------------------------------------------------------------
# Tissue normalization
# ---------------------------------------------------------------------------

class TestNormalizeTissue:
    def test_known_alias(self):
        assert atlas._normalize_tissue("crc") == "large intestine"
        assert atlas._normalize_tissue("colon") == "large intestine"
        assert atlas._normalize_tissue("Colon") == "large intestine"
        assert atlas._normalize_tissue("  CRC  ") == "large intestine"
        assert atlas._normalize_tissue("large intestine") == "large intestine"
        assert atlas._normalize_tissue("large_intestine") == "large intestine"
        assert atlas._normalize_tissue("large-intestine") == "large intestine"

    def test_unknown_tissue_raises(self):
        with pytest.raises(ValueError, match="Unknown tissue type"):
            atlas._normalize_tissue("some_novel_tissue")

    def test_typo_suggests_match(self):
        with pytest.raises(ValueError, match="Did you mean") as exc_info:
            atlas._normalize_tissue("brest")
        assert "breast" in str(exc_info.value)

    def test_error_shows_valid_list(self):
        with pytest.raises(ValueError, match="Valid tissue types"):
            atlas._normalize_tissue("xyzzy")

    def test_common_tissues(self):
        assert atlas._normalize_tissue("breast") == "breast"
        assert atlas._normalize_tissue("mammary") == "breast"
        assert atlas._normalize_tissue("brain") == "brain"
        assert atlas._normalize_tissue("lung") == "lung"
        assert atlas._normalize_tissue("liver") == "liver"
        assert atlas._normalize_tissue("kidney") == "kidney"
        assert atlas._normalize_tissue("pancreas") == "pancreas"


# ---------------------------------------------------------------------------
# Organism normalization
# ---------------------------------------------------------------------------

class TestNormalizeOrganism:
    def test_known_aliases(self):
        assert atlas._normalize_organism("human") == "homo_sapiens"
        assert atlas._normalize_organism("homo sapiens") == "homo_sapiens"
        assert atlas._normalize_organism("homo_sapiens") == "homo_sapiens"
        assert atlas._normalize_organism("homosapiense") == "homo_sapiens"
        assert atlas._normalize_organism("mouse") == "mus_musculus"
        assert atlas._normalize_organism("mus musculus") == "mus_musculus"

    def test_typo_suggests_match(self):
        with pytest.raises(ValueError, match="Did you mean"):
            atlas._normalize_organism("homo sapens")

    def test_unknown_raises_with_valid_list(self):
        with pytest.raises(ValueError, match="Valid organisms"):
            atlas._normalize_organism("alien")


# ---------------------------------------------------------------------------
# Immune-only detection
# ---------------------------------------------------------------------------

class TestImmuneOnlyGuess:
    def test_immune_only(self):
        immune_types = ["T cell", "B cell", "NK cell", "Monocyte", "Macrophage"]
        assert atlas._immune_only_guess(immune_types) is True

    def test_mixed(self):
        mixed = ["T cell", "B cell", "Epithelial cell", "Fibroblast"]
        assert atlas._immune_only_guess(mixed) is False

    def test_empty(self):
        assert atlas._immune_only_guess([]) is False

    def test_no_immune_keywords(self):
        assert atlas._immune_only_guess(["Epithelial", "Fibroblast"]) is False

    def test_non_immune_only(self):
        # Only non-immune keywords → not immune-only (no immune evidence)
        assert atlas._immune_only_guess(["Neuron", "Astrocyte"]) is False


# ---------------------------------------------------------------------------
# var_names normalization
# ---------------------------------------------------------------------------

class TestVarNamePromotion:
    def test_promotes_feature_name_to_var_names(self):
        adata = AnnData(
            X=np.zeros((2, 4), dtype=np.float32),
            obs=pd.DataFrame(index=["cell1", "cell2"]),
            var=pd.DataFrame(
                {"feature_name": ["G1", "G2", "", "G1"]},
                index=["0", "1", "2", "3"],
            ),
        )

        changed = atlas._set_var_names_from_feature_column(
            adata,
            feature_column="feature_name",
        )

        assert changed is True
        assert list(adata.var_names) == ["G1", "G2", "2", "G1-1"]
        assert adata.var_names.is_unique

    def test_noop_when_feature_column_missing(self):
        adata = AnnData(
            X=np.zeros((1, 2), dtype=np.float32),
            obs=pd.DataFrame(index=["cell1"]),
            var=pd.DataFrame(index=["0", "1"]),
        )
        original = list(adata.var_names)

        changed = atlas._set_var_names_from_feature_column(
            adata,
            feature_column="feature_name",
        )

        assert changed is False
        assert list(adata.var_names) == original


# ---------------------------------------------------------------------------
# Cell-type category capping
# ---------------------------------------------------------------------------

class TestCapCellTypeCounts:
    def test_no_cap_returns_original(self):
        counts = [("A", 10), ("B", 7), ("C", 2)]
        kept, display = atlas._cap_cell_type_counts(counts, None)
        assert kept == ["A", "B", "C"]
        assert display == counts

    def test_cap_merges_tail_into_other(self):
        counts = [("A", 10), ("B", 7), ("C", 5), ("D", 3)]
        kept, display = atlas._cap_cell_type_counts(counts, 3)
        assert kept == ["A", "B"]
        assert display == [("A", 10), ("B", 7), ("Other", 8)]

    def test_cap_one_is_only_other(self):
        counts = [("A", 10), ("B", 7), ("C", 5)]
        kept, display = atlas._cap_cell_type_counts(counts, 1)
        assert kept == []
        assert display == [("Other", 22)]

    def test_invalid_cap_raises(self):
        with pytest.raises(ValueError, match="max_cell_types must be >= 1"):
            atlas._cap_cell_type_counts([("A", 1)], 0)

    def test_cap_without_other_keeps_exact_top_n(self):
        counts = [("A", 10), ("B", 7), ("C", 5), ("D", 3)]
        kept, display = atlas._cap_cell_type_counts(counts, 3, include_other=False)
        assert kept == ["A", "B", "C"]
        assert display == [("A", 10), ("B", 7), ("C", 5)]


# ---------------------------------------------------------------------------
# Cell-type coarse mapping
# ---------------------------------------------------------------------------

class TestCellTypeMapping:
    def test_load_mapping_with_named_columns(self, tmp_path):
        mapping_file = tmp_path / "map.csv"
        mapping_file.write_text(
            "cell_type,coarse_cell_type\n"
            "T cell,Immune\n"
            "B cell,Immune\n"
            "Enterocyte,Epithelial\n",
            encoding="utf-8",
        )
        mapping = atlas._load_cell_type_mapping(mapping_file)
        assert mapping["T cell"] == "Immune"
        assert mapping["Enterocyte"] == "Epithelial"

    def test_load_mapping_uses_first_two_columns_as_fallback(self, tmp_path):
        mapping_file = tmp_path / "map.tsv"
        mapping_file.write_text(
            "fine_label\tbroad_label\n"
            "Fibroblast\tStromal\n",
            encoding="utf-8",
        )
        mapping = atlas._load_cell_type_mapping(mapping_file)
        assert mapping == {"Fibroblast": "Stromal"}

    def test_apply_mapping_defaults_unmapped_to_other(self):
        pd = pytest.importorskip("pandas")
        series = pd.Series(["T cell", "Unknown subtype", None], dtype="string")
        mapping = {"T cell": "Immune"}
        out = atlas._apply_cell_type_mapping(series, mapping)
        assert out.tolist() == ["Immune", "Other", "Other"]

    def test_auto_coarse_mapping_keywords(self):
        assert atlas._auto_coarse_cell_type_label("CD4-positive, alpha-beta T cell") == "T/NK/ILC"
        assert atlas._auto_coarse_cell_type_label("alveolar macrophage") == "Myeloid"
        assert atlas._auto_coarse_cell_type_label("capillary endothelial cell") == "Endothelial"
        assert atlas._auto_coarse_cell_type_label("transit amplifying cell") == "Stem/Progenitor"
        assert atlas._auto_coarse_cell_type_label("goblet cell") == "Epithelial"
        assert atlas._auto_coarse_cell_type_label("tuft cell of colon") == "Epithelial"
        assert atlas._auto_coarse_cell_type_label("unknown") == "Unknown"

    def test_cell_type_drop_mask_drops_unknown_and_other(self):
        pd = pytest.importorskip("pandas")
        series = pd.Series(["Unknown", "Other", "T cell"], dtype="string")
        mask = atlas._cell_type_drop_mask(
            series,
            drop_unknown=True,
            drop_other=True,
        )
        assert mask.tolist() == [True, True, False]

    def test_filter_cell_types_by_min_count(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "soma_joinid": [1, 2, 3, 4, 5, 6],
                "cell_type": ["A", "A", "A", "B", "B", "C"],
            }
        )
        out, dropped = atlas._filter_cell_types_by_min_count(
            df,
            min_cells_per_type=3,
        )
        assert set(out["cell_type"].unique().tolist()) == {"A"}
        assert dropped == [("B", 2), ("C", 1)]


# ---------------------------------------------------------------------------
# resolve_reference
# ---------------------------------------------------------------------------

class TestResolveReference:
    def test_mutual_exclusion(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            atlas.resolve_reference(
                tissue_type="colon",
                scrna_reference_path=Path("/some/ref.h5ad"),
            )

    def test_neither_provided(self):
        path, col = atlas.resolve_reference(
            tissue_type=None,
            scrna_reference_path=None,
        )
        assert path is None
        assert col == "cell_type"

    def test_explicit_path(self):
        p = Path("/my/reference.h5ad")
        path, col = atlas.resolve_reference(
            tissue_type=None,
            scrna_reference_path=p,
            scrna_celltype_column="my_col",
        )
        assert path == p
        assert col == "my_col"

    def test_tissue_type_calls_fetch(self):
        fake_ref = atlas.AtlasReference(
            h5ad_path=Path("/cache/colon.h5ad"),
            metadata_path=Path("/cache/colon.json"),
            tissue="large intestine",
            organism="homo_sapiens",
            census_version="stable",
            n_obs=1000,
            n_cell_types=10,
            cell_type_column="cell_type",
            immune_only=False,
        )
        with mock.patch.object(atlas, "fetch_reference", return_value=fake_ref):
            path, col = atlas.resolve_reference(
                tissue_type="colon", scrna_reference_path=None
            )
        assert path == Path("/cache/colon.h5ad")
        assert col == "cell_type"


# ---------------------------------------------------------------------------
# Cached reference detection (no Census)
# ---------------------------------------------------------------------------

class TestFetchReferenceCached:
    def test_uses_cache_when_present(self, tmp_path):
        """If h5ad + json already exist, fetch_reference should not call Census."""
        tissue = "breast"
        normalized = atlas._normalize_tissue(tissue)
        safe = normalized.replace(" ", "_")
        tissue_dir = tmp_path / "homo_sapiens" / safe
        tissue_dir.mkdir(parents=True)

        h5ad = tissue_dir / f"{safe}.h5ad"
        h5ad.write_bytes(b"fake h5ad content")

        meta = tissue_dir / f"{safe}.json"
        meta.write_text(json.dumps({
            "h5ad_path": str(h5ad),
            "tissue": normalized,
            "organism": "homo_sapiens",
            "census_version": "stable",
            "n_obs": 500,
            "n_cell_types": 8,
            "cell_type_column": "cell_type",
            "immune_only": False,
            "coarse_cell_types": False,
            "exclude_unknown": False,
            "effective_exclude_unknown": False,
            "drop_unknown_cell_types": False,
            "drop_other_cell_types": False,
            "metadata_prefiltered": False,
        }), encoding="utf-8")

        # Patch _require_census so we know it isn't called
        with mock.patch.object(atlas, "_require_census") as mock_census:
            ref = atlas.fetch_reference(tissue, cache_dir=tmp_path)

        mock_census.assert_not_called()
        assert ref.h5ad_path == h5ad
        assert ref.tissue == normalized
        assert ref.n_obs == 500

    def test_organism_alias_hits_same_cache(self, tmp_path):
        tissue = "breast"
        normalized = atlas._normalize_tissue(tissue)
        safe = normalized.replace(" ", "_")
        tissue_dir = tmp_path / "homo_sapiens" / safe
        tissue_dir.mkdir(parents=True)

        h5ad = tissue_dir / f"{safe}.h5ad"
        h5ad.write_bytes(b"fake h5ad content")
        meta = tissue_dir / f"{safe}.json"
        meta.write_text(json.dumps({
            "h5ad_path": str(h5ad),
            "tissue": normalized,
            "organism": "homo_sapiens",
            "census_version": "stable",
            "n_obs": 500,
            "n_cell_types": 8,
            "cell_type_column": "cell_type",
            "immune_only": False,
            "coarse_cell_types": False,
            "exclude_unknown": False,
            "effective_exclude_unknown": False,
            "drop_unknown_cell_types": False,
            "drop_other_cell_types": False,
            "metadata_prefiltered": False,
        }), encoding="utf-8")

        with mock.patch.object(atlas, "_require_census") as mock_census:
            ref = atlas.fetch_reference(tissue, organism="human", cache_dir=tmp_path)

        mock_census.assert_not_called()
        assert ref.h5ad_path == h5ad
        assert ref.organism == "homo_sapiens"

    def test_cache_rebuilds_when_mapping_option_changes(self, tmp_path):
        tissue = "breast"
        normalized = atlas._normalize_tissue(tissue)
        safe = normalized.replace(" ", "_")
        tissue_dir = tmp_path / "homo_sapiens" / safe
        tissue_dir.mkdir(parents=True)

        h5ad = tissue_dir / f"{safe}.h5ad"
        h5ad.write_bytes(b"fake h5ad content")

        meta = tissue_dir / f"{safe}.json"
        meta.write_text(json.dumps({
            "h5ad_path": str(h5ad),
            "tissue": normalized,
            "organism": "homo_sapiens",
            "census_version": "stable",
            "n_obs": 500,
            "n_cell_types": 8,
            "cell_type_column": "cell_type",
            "immune_only": False,
            "max_cells_per_type": 1000,
            "min_cells_per_type": 1,
            "max_cell_types": None,
            "coarse_cell_types": False,
            "exclude_unknown": False,
            "effective_exclude_unknown": False,
            "drop_unknown_cell_types": False,
            "drop_other_cell_types": False,
            "metadata_prefiltered": False,
            "cell_type_map_path": None,
        }), encoding="utf-8")

        mapping_file = tmp_path / "coarse_map.csv"
        mapping_file.write_text(
            "cell_type,coarse\nT cell,Immune\n",
            encoding="utf-8",
        )

        with mock.patch.object(atlas, "_require_census", side_effect=RuntimeError("rebuild")) as mock_census:
            with pytest.raises(RuntimeError, match="rebuild"):
                atlas.fetch_reference(
                    tissue,
                    cache_dir=tmp_path,
                    cell_type_map_path=mapping_file,
                )

        mock_census.assert_called_once()


# ---------------------------------------------------------------------------
# Missing Census raises ImportError
# ---------------------------------------------------------------------------

class TestFetchReferenceNoCensus:
    def test_import_error_when_not_installed(self, tmp_path):
        """fetch_reference should raise ImportError when Census is missing."""
        with mock.patch(
            "segger.utils.optional_deps.CELLXGENE_CENSUS_AVAILABLE", False
        ):
            with pytest.raises(ImportError, match="cellxgene-census"):
                atlas.fetch_reference("colon", cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# Census opener compatibility
# ---------------------------------------------------------------------------

class TestOpenCensusCompatibility:
    def test_prefers_open_soma_when_available(self):
        events: list[str] = []

        class DummyCtx:
            def __enter__(self):
                events.append("enter")
                return "ctx"

            def __exit__(self, exc_type, exc, tb):
                events.append("exit")
                return False

        class CensusModule:
            def open_soma(self, *, census_version):
                events.append(f"open_soma:{census_version}")
                return DummyCtx()

            def open(self, *, census_version):  # pragma: no cover - should not be used
                events.append(f"open:{census_version}")
                return DummyCtx()

        with atlas._open_census(CensusModule(), census_version="stable") as handle:
            assert handle == "ctx"

        assert events == ["open_soma:stable", "enter", "exit"]

    def test_falls_back_to_open_when_open_soma_missing(self):
        events: list[str] = []

        class DummyCtx:
            def __enter__(self):
                events.append("enter")
                return "ctx"

            def __exit__(self, exc_type, exc, tb):
                events.append("exit")
                return False

        class CensusModule:
            def open(self, *, census_version):
                events.append(f"open:{census_version}")
                return DummyCtx()

        with atlas._open_census(CensusModule(), census_version="latest") as handle:
            assert handle == "ctx"

        assert events == ["open:latest", "enter", "exit"]

    def test_raises_when_no_supported_open_function(self):
        class CensusModule:
            pass

        with pytest.raises(AttributeError, match="open_soma or open"):
            atlas._open_census(CensusModule(), census_version="stable")


class TestGetAnndataCompatibility:
    def test_uses_new_obs_var_column_names_api(self):
        calls: list[tuple[str, dict]] = []

        class CensusModule:
            def get_anndata(self, handle, **kwargs):
                calls.append(("new", kwargs))
                return "ok-new"

        out = atlas._get_anndata_compat(
            CensusModule(),
            object(),
            organism="homo_sapiens",
            obs_value_filter="x == y",
        )
        assert out == "ok-new"
        assert len(calls) == 1
        kwargs = calls[0][1]
        assert "obs_column_names" in kwargs
        assert "var_column_names" in kwargs
        assert "column_names" not in kwargs

    def test_falls_back_to_column_names_on_type_error(self):
        calls: list[dict] = []

        class CensusModule:
            def __init__(self):
                self.first = True

            def get_anndata(self, handle, **kwargs):
                calls.append(kwargs)
                if self.first:
                    self.first = False
                    raise TypeError("unexpected keyword argument 'obs_column_names'")
                return "ok-old"

        out = atlas._get_anndata_compat(
            CensusModule(),
            object(),
            organism="homo_sapiens",
            obs_value_filter="x == y",
        )
        assert out == "ok-old"
        assert len(calls) == 2
        assert "obs_column_names" in calls[0]
        assert "column_names" in calls[1]


class TestMetadataFirstFetch:
    def test_fetch_requires_get_obs_metadata(self, tmp_path):
        class _DummyCtx:
            def __enter__(self):
                return object()

            def __exit__(self, exc_type, exc, tb):
                return False

        class CensusModule:
            def open_soma(self, *, census_version):
                return _DummyCtx()

            # Deliberately no get_obs() to trigger metadata-first guard.
            def get_anndata(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("Should not call get_anndata without metadata prefilter")

        with mock.patch.object(atlas, "_require_census", return_value=CensusModule()):
            with pytest.raises(RuntimeError, match="Metadata-first atlas fetch requires"):
                atlas.fetch_reference(
                    "breast",
                    cache_dir=tmp_path,
                    force=True,
                    coarse_cell_types=True,
                )


# ---------------------------------------------------------------------------
# list_cached_references
# ---------------------------------------------------------------------------

class TestListCachedReferences:
    def test_empty_cache(self, tmp_path):
        refs = atlas.list_cached_references(cache_dir=tmp_path)
        assert refs == []

    def test_finds_references(self, tmp_path):
        for tissue in ("breast", "brain"):
            d = tmp_path / "homo_sapiens" / tissue
            d.mkdir(parents=True)
            (d / f"{tissue}.h5ad").write_bytes(b"fake")
            (d / f"{tissue}.json").write_text(json.dumps({
                "h5ad_path": str(d / f"{tissue}.h5ad"),
                "tissue": tissue,
                "organism": "homo_sapiens",
                "census_version": "stable",
                "n_obs": 100,
                "n_cell_types": 5,
                "cell_type_column": "cell_type",
                "immune_only": False,
            }), encoding="utf-8")

        refs = atlas.list_cached_references(cache_dir=tmp_path)
        assert len(refs) == 2
        tissues = {r.tissue for r in refs}
        assert tissues == {"brain", "breast"}


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------

class TestClearCache:
    def test_clear_specific_tissue(self, tmp_path):
        d = tmp_path / "homo_sapiens" / "breast"
        d.mkdir(parents=True)
        (d / "breast.h5ad").write_bytes(b"fake")

        removed = atlas.clear_cache(tissue="breast", cache_dir=tmp_path)
        assert removed == 1
        assert not d.exists()

    def test_clear_all(self, tmp_path):
        for tissue in ("breast", "brain"):
            d = tmp_path / "homo_sapiens" / tissue
            d.mkdir(parents=True)
            (d / f"{tissue}.h5ad").write_bytes(b"fake")

        removed = atlas.clear_cache(tissue=None, cache_dir=tmp_path)
        assert removed == 2

    def test_clear_nonexistent(self, tmp_path):
        removed = atlas.clear_cache(tissue="nonexistent", cache_dir=tmp_path)
        assert removed == 0


# ---------------------------------------------------------------------------
# AtlasReference dataclass
# ---------------------------------------------------------------------------

class TestAtlasReference:
    def test_frozen(self):
        ref = atlas.AtlasReference(
            h5ad_path=Path("/x.h5ad"),
            metadata_path=Path("/x.json"),
            tissue="breast",
            organism="homo_sapiens",
            census_version="stable",
            n_obs=100,
            n_cell_types=5,
            cell_type_column="cell_type",
            immune_only=False,
        )
        with pytest.raises(AttributeError):
            ref.tissue = "brain"  # type: ignore[misc]
