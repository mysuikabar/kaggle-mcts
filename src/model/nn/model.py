from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from typing_extensions import Self

from ..base import BaseModel
from .dataset import MCTSDataModule, MCTSDataset
from .module import NNModule
from .processor import Preprocessor

NUM_WORKERS = 0  # os.cpu_count()


class NNModel(BaseModel):
    def __init__(
        self,
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
        self._model: NNModule | None = None
        self._processor = Preprocessor()
        self._categorical_feature_dims = categorical_feature_dims.copy()
        self._categorical_features = list(categorical_feature_dims.keys())
        self._embedding_dim = embedding_dim
        self._hidden_dims = hidden_dims
        self._dropout_rate = dropout_rate
        self._learning_rate = learning_rate
        self._scheduler_patience = scheduler_patience
        self._max_epochs = max_epochs
        self._early_stopping_patience = early_stopping_patience
        self._batch_size = batch_size

    def fit(self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> Self:
        # data
        X_tr = self._processor.fit_transform(X_tr, self._categorical_feature_dims)
        X_va = self._processor.transform(X_va)

        self._data_module = MCTSDataModule(
            X_tr=X_tr,
            X_va=X_va,
            y_tr=y_tr,
            y_va=y_va,
            categorical_features=self._categorical_features,
            batch_size=self._batch_size,
        )

        # model
        self._model = NNModule(
            num_numerical_features=X_tr.shape[1] - len(self._categorical_features),
            categorical_feature_dims=self._categorical_feature_dims,
            embedding_dim=self._embedding_dim,
            hidden_dims=self._hidden_dims,
            dropout_rate=self._dropout_rate,
            learning_rate=self._learning_rate,
            scheduler_patience=self._scheduler_patience,
        )

        # training
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=self._early_stopping_patience)
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
        self._trainer = pl.Trainer(
            max_epochs=self._max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
        )
        self._trainer.fit(self._model, datamodule=self._data_module)

        # load best model
        self._model = NNModule.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            num_numerical_features=X_tr.shape[1] - len(self._categorical_features),
            categorical_feature_dims=self._categorical_feature_dims,
            embedding_dim=self._embedding_dim,
            hidden_dims=self._hidden_dims,
            dropout_rate=self._dropout_rate,
            learning_rate=self._learning_rate,
            scheduler_patience=self._scheduler_patience,
        ).to("cpu")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fitted before prediction")

        X = self._processor.transform(X)
        dataset = MCTSDataset(X, self._categorical_features)
        dataloader = DataLoader(dataset, batch_size=self._batch_size, num_workers=NUM_WORKERS)

        preds = []
        self._model.eval()
        with torch.no_grad():
            for batch in dataloader:
                numerical = batch["numerical"]
                categorical = batch["categorical"]
                pred = self._model(numerical, categorical)
                preds.append(pred.cpu())

        return torch.cat(preds).numpy()

    def save(self, filepath: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("Model must be fitted before saving")
        torch.save(self._model.state_dict(), filepath)

    def load(self, filepath: str | Path) -> Self:  # type: ignore
        raise NotImplementedError  # TODO: implement
