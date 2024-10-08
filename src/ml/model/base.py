import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from typing_extensions import Self

from ..utils import to_dict


@dataclass
class BaseConfig:
    pass


class BaseModel(ABC):
    def __init__(self, config: BaseConfig) -> None:
        self._params = to_dict(config)

    @abstractmethod
    def fit(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray
    ) -> Self:
        return self

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
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
