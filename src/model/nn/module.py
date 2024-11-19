from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam  # TODO: use AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class NNModule(pl.LightningModule):
    def __init__(
        self,
        num_numerical_features: int,
        categorical_feature_dims: dict[str, int],
        embedding_dim: int,
        hidden_dims: list[int],
        dropout_rate: float,
        learning_rate: float,
        scheduler_patience: int,
    ) -> None:
        super().__init__()

        # embedding layers
        self._embedding_layers = nn.ModuleDict(
            {
                feature: nn.Embedding(num_embeddings + 1, embedding_dim)  # +1 for unknown values
                for feature, num_embeddings in categorical_feature_dims.items()
            }
        )

        # MLP layers
        layers = []
        input_dim = num_numerical_features + embedding_dim * len(categorical_feature_dims)
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

        # hyperparameters
        self._learning_rate = learning_rate
        self._scheduler_patience = scheduler_patience

    def forward(self, numerical: torch.Tensor, categorical: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for feature, data in categorical.items():
            embedding = self._embedding_layers[feature](data)
            embeddings.append(embedding)

        x = torch.cat([*embeddings, numerical], dim=1)
        return self._layers(x).squeeze(-1)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        output = self(batch["numerical"], batch["categorical"])
        loss = self._criterion(output, batch["target"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        output = self(batch["numerical"], batch["categorical"])
        loss = self._criterion(output, batch["target"])
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int | None = None) -> torch.Tensor:
        return self(batch["numerical"], batch["categorical"])

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore
        optimizer = Adam(self.parameters(), lr=self._learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=self._scheduler_patience, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
