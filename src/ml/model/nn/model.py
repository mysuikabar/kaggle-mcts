from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing_extensions import Self

from ..base import BaseConfig, BaseModel
from .dataset import EvalTabularDataset, TrainingTabularDataset

NUM_WORKERS = 4  # os.cpu_count()


@dataclass
class NNConfig(BaseConfig):
    categorical_feature_dims: dict[str, int]
    embedding_dim: int
    hidden_dims: list[int]
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    max_epochs: int = 3000
    early_stopping_patience: int = 10
    batch_size: int = 64


class RegressionModel(pl.LightningModule):
    def __init__(
        self,
        num_numerical_features: int,
        categorical_feature_dims: dict[str, int],
        embedding_dim: int,
        hidden_dims: list[int],
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
    ) -> None:
        super().__init__()

        # embedding layers
        self._embedding_layers = nn.ModuleDict(
            {
                feature: nn.Embedding(num_embeddings, embedding_dim)
                for feature, num_embeddings in categorical_feature_dims.items()
            }
        )

        # MLP layers
        layers = []
        input_dim = num_numerical_features + embedding_dim * len(
            categorical_feature_dims
        )
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self._layers = nn.Sequential(*layers)

        # loss function
        self._criterion = nn.MSELoss()

        self._learning_rate = learning_rate

    def forward(
        self, numerical: torch.Tensor, categorical: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        for feature, data in categorical.items():
            embedding = self._embedding_layers[feature](data)
            embeddings.append(embedding)

        x = torch.cat([*embeddings, numerical], dim=1)
        return self._layers(x).squeeze(-1)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        output = self(batch["numerical"], batch["categorical"])
        loss = self._criterion(output, batch["target"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        output = self(batch["numerical"], batch["categorical"])
        loss = self._criterion(output, batch["target"])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)


class NNModel(BaseModel):
    def __init__(self, config: NNConfig) -> None:
        super().__init__(config)
        self._model: RegressionModel | None = None
        self._categorical_features = list(config.categorical_feature_dims.keys())

    def fit(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray
    ) -> Self:
        # setup datasets and dataloaders
        dataset_tr = TrainingTabularDataset(
            X=X_tr,
            categorical_features=self._categorical_features,
            y=y_tr,
        )
        self._scaler = dataset_tr.fitted_scaler
        self._encoders = dataset_tr.fitted_encoders

        dataset_va = EvalTabularDataset(
            X=X_va,
            categorical_features=self._categorical_features,
            scaler=self._scaler,
            encoders=self._encoders,
            y=y_va,
        )

        loader_tr = DataLoader(
            dataset_tr,
            batch_size=self._params["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        loader_va = DataLoader(
            dataset_va,
            batch_size=self._params["batch_size"],
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        # initialize model
        num_numerical_features = len(X_tr.columns) - len(self._categorical_features)
        categorical_feature_dims = {
            feature: len(encoder.classes_)
            for feature, encoder in self._encoders.items()
        }

        self._model = RegressionModel(
            num_numerical_features=num_numerical_features,
            categorical_feature_dims=categorical_feature_dims,
            embedding_dim=self._params["embedding_dim"],
            hidden_dims=self._params["hidden_dims"],
            dropout_rate=self._params["dropout_rate"],
        )

        # training
        trainer = pl.Trainer(
            max_epochs=self._params["max_epochs"],
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss", patience=self._params["early_stopping_patience"]
                )
            ],
        )
        trainer.fit(self._model, loader_tr, loader_va)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fitted before prediction")

        # setup dataset and dataloader
        dataset = EvalTabularDataset(
            X=X,
            categorical_features=self._categorical_features,
            scaler=self._scaler,
            encoders=self._encoders,
        )
        loader = DataLoader(
            dataset, batch_size=self._params["batch_size"], shuffle=False
        )

        # prediction
        self._model.eval()
        predictions = []
        with torch.no_grad():
            for batch in loader:
                output = self._model(batch["numerical"], batch["categorical"])
                predictions.append(output.cpu().numpy())

        return np.concatenate(predictions)
