from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset


class BaseTabularDataset(Dataset):
    def __init__(
        self,
        X: pd.DataFrame,
        categorical_features: list[str],
        scaler: StandardScaler,
        encoders: dict[str, LabelEncoder],
        y: np.ndarray | None = None,
    ) -> None:
        # numerical features
        self._scaler = scaler
        numeric_features = [col for col in X.columns if col not in categorical_features]
        scaled = self._scaler.transform(X[numeric_features])
        self._numerical = torch.tensor(scaled, dtype=torch.float32)

        # categorical features
        self._categorical = {}
        self._encoders = encoders
        for feature in categorical_features:
            known_categories = set(self._encoders[feature].classes_)
            X_processed = X[feature].map(
                lambda x: x if x in known_categories else "<UNK>"
            )
            encoded = self._encoders[feature].transform(X_processed)
            self._categorical[feature] = torch.tensor(encoded, dtype=torch.long)

        # target
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


class TrainingTabularDataset(BaseTabularDataset):
    def __init__(
        self,
        X: pd.DataFrame,
        categorical_features: list[str],
        y: np.ndarray | None = None,
    ) -> None:
        numeric_features = [col for col in X.columns if col not in categorical_features]
        scaler = StandardScaler().fit(X[numeric_features])

        encoders = {}
        for feature in categorical_features:
            unique_values = X[feature].unique().tolist()
            unique_values.append("<UNK>")
            encoders[feature] = LabelEncoder().fit(unique_values)

        super().__init__(X, categorical_features, scaler, encoders, y)

    @property
    def fitted_scaler(self) -> StandardScaler:
        return self._scaler

    @property
    def fitted_encoders(self) -> dict[str, LabelEncoder]:
        return self._encoders


class EvalTabularDataset(BaseTabularDataset):
    def __init__(
        self,
        X: pd.DataFrame,
        categorical_features: list[str],
        scaler: StandardScaler,
        encoders: dict[str, LabelEncoder],
        y: np.ndarray | None = None,
    ) -> None:
        super().__init__(X, categorical_features, scaler, encoders, y)
