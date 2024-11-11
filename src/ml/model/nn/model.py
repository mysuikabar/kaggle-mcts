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
from .dataset import MCTSDataModule, MCTSDataset
from .processor import Preprocessor

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


class NNModule(pl.LightningModule):
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
                feature: nn.Embedding(
                    num_embeddings + 1, embedding_dim
                )  # +1 for unknown values
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
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
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
        self._model: NNModule | None = None
        self._processor = Preprocessor()
        self._categorical_feature_dims = config.categorical_feature_dims
        self._categorical_features = list(config.categorical_feature_dims.keys())

    def fit(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray
    ) -> Self:
        X_tr = self._processor.fit_transform(X_tr, self._categorical_feature_dims)
        X_va = self._processor.transform(X_va)

        self._data_module = MCTSDataModule(
            X_tr=X_tr,
            X_va=X_va,
            y_tr=y_tr,
            y_va=y_va,
            categorical_features=self._categorical_features,
            batch_size=self._params["batch_size"],
        )

        self._model = NNModule(
            num_numerical_features=X_tr.shape[1] - len(self._categorical_features),
            categorical_feature_dims=self._params["categorical_feature_dims"],
            embedding_dim=self._params["embedding_dim"],
            hidden_dims=self._params["hidden_dims"],
            dropout_rate=self._params["dropout_rate"],
            learning_rate=self._params["learning_rate"],
        )

        self._trainer = pl.Trainer(
            max_epochs=self._params["max_epochs"],
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss", patience=self._params["early_stopping_patience"]
                )
            ],
            accelerator="cpu",
        )

        self._trainer.fit(self._model, datamodule=self._data_module)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fitted before prediction")

        X = self._processor.transform(X)
        dataset = MCTSDataset(X, self._categorical_features)
        dataloader = DataLoader(
            dataset, batch_size=self._params["batch_size"], num_workers=NUM_WORKERS
        )

        preds = []
        self._model.eval()
        with torch.no_grad():
            for batch in dataloader:
                numerical = batch["numerical"]
                categorical = batch["categorical"]
                pred = self._model(numerical, categorical)
                preds.append(pred)

        return torch.cat(preds).numpy()
