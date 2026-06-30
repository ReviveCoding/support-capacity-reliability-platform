from __future__ import annotations

import builtins

import pytest

from support_capacity_reliability.config import load_config
from support_capacity_reliability.forecasting import factory


def test_base_smoke_config_does_not_require_torch():
    config = load_config("configs/smoke.yaml")
    assert "torch_quantile" not in config.forecast.models


def test_torch_model_has_clear_optional_dependency_error(monkeypatch):
    config = load_config("configs/smoke_torch.yaml")
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "support_capacity_reliability.forecasting.torch_model":
            exc = ModuleNotFoundError("No module named 'torch'")
            exc.name = "torch"
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match=r"support-capacity-reliability\[torch\]"):
        factory.build_model("torch_quantile", config)


def test_pipeline_import_does_not_eagerly_import_torch(monkeypatch):
    import importlib
    import sys

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            exc = ModuleNotFoundError("No module named 'torch'")
            exc.name = "torch"
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    sys.modules.pop("support_capacity_reliability.pipeline", None)
    module = importlib.import_module("support_capacity_reliability.pipeline")
    assert callable(module.run_pipeline)
