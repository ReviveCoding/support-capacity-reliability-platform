from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from support_capacity_reliability.forecasting.base import ForecastOutput, enforce_quantile_order


class QuantileNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=prediction.device, dtype=prediction.dtype)
    error = target[:, None] - prediction
    return torch.maximum(quantiles * error, (quantiles - 1) * error).mean()


class TorchQuantileForecaster:
    name = "torch_quantile"

    def __init__(
        self,
        epochs: int = 8,
        hidden_dim: int = 64,
        batch_size: int = 256,
        learning_rate: float = 0.003,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model: QuantileNetwork | None = None
        self.training_history: list[float] = []

    def fit(self, frame: pd.DataFrame, features: list[str], target: str) -> TorchQuantileForecaster:
        torch.manual_seed(self.seed)
        if self.device == "cpu":
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        x = self.imputer.fit_transform(frame[features])
        x = self.scaler.fit_transform(x).astype("float32")
        y = np.clip(frame[target].to_numpy(float), 0, None).astype("float32")
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
        self.model = QuantileNetwork(x.shape[1], self.hidden_dim).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        self.training_history = []
        self.model.train()
        for _ in range(self.epochs):
            losses: list[float] = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(batch_x)
                loss = pinball_loss(prediction, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self.training_history.append(float(np.mean(losses)))
        return self

    def predict(self, frame: pd.DataFrame, features: list[str]) -> ForecastOutput:
        if self.model is None:
            raise RuntimeError("Torch model is not fitted")
        x = self.imputer.transform(frame[features])
        x = self.scaler.transform(x).astype("float32")
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()
        prediction = np.clip(prediction, 0, None)
        q10, q50, q90 = enforce_quantile_order(prediction[:, 0], prediction[:, 1], prediction[:, 2])
        return ForecastOutput(self.name, q10, q50, q90)

    def hardware_summary(self) -> dict[str, object]:
        return {
            "requested_device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "cuda_device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "epochs": self.epochs,
            "training_history": self.training_history,
        }
