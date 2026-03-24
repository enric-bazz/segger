from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types

import pytest

pytest.importorskip("cyclopts")

from segger.cli import main as cli_main


def _install_fake_predict_runtime(monkeypatch: pytest.MonkeyPatch, datamodule_hparams: dict) -> dict:
    captured: dict[str, object] = {}

    @dataclass
    class FakeISTDataModule:
        input_directory: Path
        prediction_graph_scale_factor: float = 1.2
        use_3d: bool | str = False

        def __post_init__(self) -> None:
            captured["datamodule_kwargs"] = {
                "input_directory": self.input_directory,
                "prediction_graph_scale_factor": self.prediction_graph_scale_factor,
                "use_3d": self.use_3d,
            }
            self.ad = types.SimpleNamespace(shape=(1, 1))

    class FakeWriter:
        def __init__(self, output_directory: Path, **kwargs) -> None:
            captured["writer"] = {
                "output_directory": output_directory,
                **kwargs,
            }

    class FakeTrainer:
        def __init__(self, **kwargs) -> None:
            captured["trainer_init"] = kwargs

        def predict(self, **kwargs) -> None:
            captured["trainer_predict"] = kwargs

    class FakeSLURMEnvironment:
        detect = staticmethod(lambda: True)

    class FakeModel:
        def __init__(self) -> None:
            self.hparams = {}

    class FakeLitISTEncoder:
        @staticmethod
        def load_from_checkpoint(checkpoint_path: Path, map_location: str = "cpu") -> FakeModel:
            captured["load_from_checkpoint"] = {
                "checkpoint_path": checkpoint_path,
                "map_location": map_location,
            }
            return FakeModel()

    fake_data_module = types.ModuleType("segger.data")
    fake_data_module.ISTDataModule = FakeISTDataModule
    fake_data_module.ISTSegmentationWriter = FakeWriter

    fake_models_module = types.ModuleType("segger.models")
    fake_models_module.LitISTEncoder = FakeLitISTEncoder

    lightning_module = types.ModuleType("lightning")
    lightning_pytorch_module = types.ModuleType("lightning.pytorch")
    lightning_plugins_module = types.ModuleType("lightning.pytorch.plugins")
    lightning_env_module = types.ModuleType("lightning.pytorch.plugins.environments")
    lightning_pytorch_module.Trainer = FakeTrainer
    lightning_env_module.SLURMEnvironment = FakeSLURMEnvironment
    lightning_plugins_module.environments = lightning_env_module
    lightning_pytorch_module.plugins = lightning_plugins_module
    lightning_module.pytorch = lightning_pytorch_module

    monkeypatch.setitem(sys.modules, "segger.data", fake_data_module)
    monkeypatch.setitem(sys.modules, "segger.models", fake_models_module)
    monkeypatch.setitem(sys.modules, "lightning", lightning_module)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", lightning_pytorch_module)
    monkeypatch.setitem(sys.modules, "lightning.pytorch.plugins", lightning_plugins_module)
    monkeypatch.setitem(sys.modules, "lightning.pytorch.plugins.environments", lightning_env_module)
    monkeypatch.setattr(
        cli_main,
        "_load_checkpoint_datamodule_hparams",
        lambda _checkpoint_path: dict(datamodule_hparams),
    )

    return captured


def test_predict_uses_checkpoint_scale_factor_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_predict_runtime(
        monkeypatch,
        {"prediction_graph_scale_factor": 2.2, "use_3d": False},
    )

    cli_main.predict(
        checkpoint_path=tmp_path / "model.ckpt",
        input_directory=tmp_path / "input",
        output_directory=tmp_path / "output",
    )

    datamodule_kwargs = captured["datamodule_kwargs"]
    assert datamodule_kwargs["prediction_graph_scale_factor"] == 2.2
    assert datamodule_kwargs["use_3d"] is False
    assert captured["trainer_predict"]["return_predictions"] is False


def test_predict_allows_prediction_scale_override_and_preserves_fragment_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_predict_runtime(
        monkeypatch,
        {"prediction_graph_scale_factor": 2.2, "use_3d": False},
    )

    cli_main.predict(
        checkpoint_path=tmp_path / "model.ckpt",
        input_directory=tmp_path / "input",
        output_directory=tmp_path / "output",
        prediction_scale_factor=3.2,
        fragment_mode=True,
        fragment_min_transcripts=7,
        fragment_similarity_threshold=0.6,
        min_similarity=0.25,
        min_similarity_shift=0.1,
    )

    datamodule_kwargs = captured["datamodule_kwargs"]
    writer_kwargs = captured["writer"]

    assert datamodule_kwargs["prediction_graph_scale_factor"] == 3.2
    assert writer_kwargs["fragment_mode"] is True
    assert writer_kwargs["fragment_min_transcripts"] == 7
    assert writer_kwargs["fragment_similarity_threshold"] == 0.6
    assert writer_kwargs["min_similarity"] == 0.25
    assert writer_kwargs["min_similarity_shift"] == 0.1


def test_predict_accepts_legacy_min_fragment_size_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_predict_runtime(
        monkeypatch,
        {"prediction_graph_scale_factor": 2.2, "use_3d": False},
    )

    cli_main.predict(
        checkpoint_path=tmp_path / "model.ckpt",
        input_directory=tmp_path / "input",
        output_directory=tmp_path / "output",
        min_fragment_size=9,
    )

    writer_kwargs = captured["writer"]
    assert writer_kwargs["fragment_min_transcripts"] == 9
