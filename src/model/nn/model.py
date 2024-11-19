from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from typing_extensions import Self

from ..base import BaseModel
from .dataset import MCTSDataModule, MCTSDataset
from .module import NNModule

NUM_WORKERS = 0  # os.cpu_count()


class NNModel(BaseModel):
    def __init__(
        self,
        num_numerical_features: int,
        categorical_feature_dims: dict[str, int],
        embedding_dim: int,
        hidden_dims: list[int],
        dropout_rate: float,
        learning_rate: float,
        scheduler_patience: int,
        max_epochs: int,
        early_stopping_patience: int,
        batch_size: int,
    ) -> None:
        self._model_params = {
            "num_numerical_features": num_numerical_features,
            "categorical_feature_dims": categorical_feature_dims,
            "embedding_dim": embedding_dim,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "scheduler_patience": scheduler_patience,
        }
        self._max_epochs = max_epochs
        self._early_stopping_patience = early_stopping_patience
        self._batch_size = batch_size

        self._model = NNModule(**self._model_params)  # type: ignore

    def fit(self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> Self:
        datamodule = MCTSDataModule(X_tr, X_va, y_tr, y_va, batch_size=self._batch_size)

        early_stop_callback = EarlyStopping(monitor="val_loss", patience=self._early_stopping_patience)
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
        trainer = Trainer(max_epochs=self._max_epochs, callbacks=[early_stop_callback, checkpoint_callback])
        trainer.fit(self._model, datamodule=datamodule)

        # load best model
        self._model = NNModule.load_from_checkpoint(checkpoint_callback.best_model_path, **self._model_params).to("cpu")  # type: ignore

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        dataset = MCTSDataset(X)
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False, num_workers=NUM_WORKERS)

        self._model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                output = self._model(batch["numerical"], batch["categorical"])
                predictions.append(output.cpu().numpy())

        return np.concatenate(predictions)

    def save(self, filepath: str | Path) -> None:
        torch.save(
            {
                "model": self._model.state_dict(),
                "model_params": self._model_params,
                "max_epochs": self._max_epochs,
                "early_stopping_patience": self._early_stopping_patience,
                "batch_size": self._batch_size,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        state = torch.load(filepath, weights_only=True)

        self = cls(
            max_epochs=state["max_epochs"],
            early_stopping_patience=state["early_stopping_patience"],
            batch_size=state["batch_size"],
            **state["model_params"],
        )
        self._model.load_state_dict(state["model"])

        return self
