import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="optional torch extra is not installed")

from support_capacity_reliability.forecasting.torch_model import (  # noqa: E402
    TorchQuantileForecaster,
    pinball_loss,
)


def test_pinball_loss_is_finite_and_differentiable():
    prediction = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], requires_grad=True)
    target = torch.tensor([2.5, 3.5])
    loss = pinball_loss(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_torch_quantile_forecaster_trains_and_orders_predictions():
    rng = np.random.default_rng(17)
    rows = 48
    frame = pd.DataFrame(
        {
            "feature_1": rng.normal(size=rows),
            "feature_2": rng.normal(size=rows),
        }
    )
    frame["target"] = np.clip(3.0 + frame["feature_1"] - 0.4 * frame["feature_2"], 0, None)
    model = TorchQuantileForecaster(
        epochs=2,
        hidden_dim=8,
        batch_size=16,
        learning_rate=0.01,
        device="cpu",
        seed=17,
    ).fit(frame, ["feature_1", "feature_2"], "target")
    output = model.predict(frame.iloc[:8], ["feature_1", "feature_2"])
    assert len(model.training_history) == 2
    assert np.isfinite(model.training_history).all()
    assert np.all(output.q10 >= 0)
    assert np.all(output.q10 <= output.q50)
    assert np.all(output.q50 <= output.q90)
    hardware = model.hardware_summary()
    assert hardware["requested_device"] == "cpu"
    assert hardware["epochs"] == 2
