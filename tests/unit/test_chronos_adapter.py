import numpy as np
import pandas as pd
import pytest

from support_capacity_reliability.forecasting.chronos_adapter import Chronos2Adapter


class FakeChronosPipeline:
    def predict_quantiles(self, contexts, prediction_length, quantile_levels):
        assert prediction_length == 1
        assert quantile_levels == [0.1, 0.5, 0.9]
        assert np.allclose(contexts[0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        return np.array([[[2.0, 3.0, 4.0]], [[4.0, 5.0, 6.0]]])


def test_chronos_adapter_uses_dense_oldest_to_newest_context():
    frame = pd.DataFrame(
        {
            "lag_8": [1.0, 2.0],
            "lag_7": [2.0, 3.0],
            "lag_6": [3.0, 4.0],
            "lag_5": [4.0, 5.0],
            "lag_4": [5.0, 6.0],
            "lag_3": [6.0, 7.0],
            "lag_2": [7.0, 8.0],
            "lag_1": [8.0, 9.0],
        }
    )
    adapter = Chronos2Adapter(context_length=8)
    adapter.pipeline = FakeChronosPipeline()
    output = adapter.predict(frame, list(frame.columns))
    assert np.allclose(output.q50, [3.0, 5.0])


def test_chronos_adapter_rejects_sparse_context():
    adapter = Chronos2Adapter(context_length=8)
    adapter.pipeline = FakeChronosPipeline()
    with pytest.raises(ValueError, match="dense lag context"):
        adapter.predict(pd.DataFrame({"lag_1": [1.0]}), ["lag_1"])
