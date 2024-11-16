from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from typing_extensions import Self


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> Self:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, filepath: str | Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str | Path) -> Self:
        pass
