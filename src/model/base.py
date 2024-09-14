import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from typing_extensions import Self


class BaseModel(ABC):
    @abstractmethod
    def fit(
        self, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray
    ) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def save(self, filepath: str | Path) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        with open(filepath, "rb") as file:
            model = pickle.load(file)

        if not isinstance(model, cls):
            raise TypeError(
                f"Loaded object type does not match expected type. "
                f"Expected: {cls.__name__}, Actual: {type(model).__name__}"
            )

        return model
