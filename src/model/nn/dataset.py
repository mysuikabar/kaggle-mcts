from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

NUM_WORKERS = 0  # os.cpu_count()


class MCTSDataset(Dataset):
    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray | None = None,
    ) -> None:
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        self._numerical = torch.tensor(X[numerical_features].values, dtype=torch.float32)
        self._categorical = {col: torch.tensor(X[col].values, dtype=torch.int64) for col in categorical_features}
        self._target = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self._numerical)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {
            "numerical": self._numerical[idx],
            "categorical": {k: v[idx] for k, v in self._categorical.items()},
        }
        if self._target is not None:
            item["target"] = self._target[idx]

        return item


class MCTSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X_tr: pd.DataFrame,
        X_va: pd.DataFrame,
        y_tr: np.ndarray,
        y_va: np.ndarray,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self._train_dataset = MCTSDataset(X_tr, y_tr)
        self._val_dataset = MCTSDataset(X_va, y_va)
        self._batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
