from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Self

from .base import BaseConfig, BaseModel


@dataclass
class NNConfig(BaseConfig):
    embedding_dims: dict[str, int]
    categorical_features: list[str]
    hidden_dims: list[int]
    dropout_rate: float = 0.1
    batch_size: int = 64
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10


class TabularDataset(Dataset):
    def __init__(
        self,
        X: pd.DataFrame,
        categorical_features: list[str],
        encoders: dict[str, LabelEncoder] | None = None,
        y: np.ndarray | None = None,
        is_training: bool = False,
    ) -> None:
        # カテゴリ特徴量以外を数値特徴量として扱う
        numeric_features: list[str] = [
            col for col in X.columns if col not in categorical_features
        ]
        self.numerical: torch.Tensor = torch.tensor(
            X[numeric_features].values, dtype=torch.float32
        )
        self.categorical: dict[str, torch.Tensor] = {}
        self.encoders: dict[str, LabelEncoder] = {} if encoders is None else encoders

        # エンコーディング処理
        for feat in categorical_features:
            if is_training:
                self.encoders[feat] = LabelEncoder()
                encoded = self.encoders[feat].fit_transform(X[feat])
            else:
                encoded = self.encoders[feat].transform(X[feat])
            self.categorical[feat] = torch.tensor(encoded, dtype=torch.long)

        self.target: torch.Tensor | None = (
            torch.tensor(y, dtype=torch.float32) if y is not None else None
        )

    def __len__(self) -> int:
        return len(self.numerical)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "numerical": self.numerical[idx],
            "categorical": {k: v[idx] for k, v in self.categorical.items()},
        }
        if self.target is not None:
            item["target"] = self.target[idx]
        return item


class RegressionModel(pl.LightningModule):
    def __init__(self, config: NNConfig, num_numerical_features: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config: NNConfig = config

        # Embedding layers
        self.embedding_layers: nn.ModuleDict = nn.ModuleDict()
        self.encoders: dict[str, LabelEncoder] = {}

        # MLP layers
        input_dim: int = num_numerical_features + sum(config.embedding_dims.values())
        layers: list[nn.Module] = []
        prev_dim: int = input_dim

        for hidden_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.layers: nn.Sequential = nn.Sequential(*layers)

        # Loss function
        self.criterion: nn.Module = nn.MSELoss()

    def setup_embeddings(self, cardinalities: dict[str, int]) -> None:
        """Embedding層の初期化（カーディナリティ確定後に呼び出し）"""
        self.embedding_layers = nn.ModuleDict(
            {
                feat_name: nn.Embedding(
                    cardinality, self.config.embedding_dims[feat_name]
                )
                for feat_name, cardinality in cardinalities.items()
            }
        )

    def forward(
        self, numerical: torch.Tensor, categorical: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Process embeddings
        embeddings: list[torch.Tensor] = []
        for feat_name, feat_values in categorical.items():
            embedding = self.embedding_layers[feat_name](feat_values)
            embeddings.append(embedding)

        # Concatenate features
        x = torch.cat([*embeddings, numerical], dim=1)
        return self.layers(x).squeeze(-1)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        output = self(batch["numerical"], batch["categorical"])
        loss = self.criterion(output, batch["target"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        output = self(batch["numerical"], batch["categorical"])
        loss = self.criterion(output, batch["target"])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)


class NNModel(BaseModel):
    def __init__(self, config: NNConfig) -> None:
        super().__init__(config)
        self._config: NNConfig = config
        self._model: RegressionModel | None = None
        self._encoders: dict[str, LabelEncoder] | None = None

    def fit(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray
    ) -> Self:
        # データセットの作成
        train_dataset = TabularDataset(
            X_tr,
            self._config.categorical_features,
            y=y_tr,
            is_training=True,
        )

        valid_dataset = TabularDataset(
            X_va,
            self._config.categorical_features,
            encoders=train_dataset.encoders,
            y=y_va,
        )

        # データローダーの作成
        train_loader = DataLoader(
            train_dataset, batch_size=self._config.batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self._config.batch_size, shuffle=False
        )

        # モデルの初期化
        num_numerical_features: int = len(X_tr.columns) - len(
            self._config.categorical_features
        )
        self._model = RegressionModel(self._config, num_numerical_features)
        self._encoders = train_dataset.encoders

        # カーディナリティの取得とEmbedding層の設定
        cardinalities: dict[str, int] = {
            feat: len(encoder.classes_) for feat, encoder in self._encoders.items()
        }
        self._model.setup_embeddings(cardinalities)

        # 学習の設定
        trainer = pl.Trainer(
            max_epochs=self._config.max_epochs,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self._config.early_stopping_patience,
                )
            ],
        )

        # 学習の実行
        trainer.fit(self._model, train_loader, valid_loader)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None or self._encoders is None:
            raise RuntimeError("Model must be fitted before prediction")

        # データセットの作成
        dataset = TabularDataset(
            X,
            self._config.categorical_features,
            encoders=self._encoders,
        )

        dataloader = DataLoader(
            dataset, batch_size=self._config.batch_size, shuffle=False
        )

        # 予測
        self._model.eval()
        predictions: list[np.ndarray] = []

        with torch.no_grad():
            for batch in dataloader:
                output = self._model(batch["numerical"], batch["categorical"])
                predictions.append(output.cpu().numpy())

        return np.concatenate(predictions)
