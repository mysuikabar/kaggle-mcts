import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from typing_extensions import Self


class BaseProcessor(ABC):
    """
    Base class for data processors
    """

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def save(self, filepath: str | Path) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        with open(filepath, "rb") as file:
            processor = pickle.load(file)

        if not isinstance(processor, cls):
            raise TypeError(
                f"Loaded object type does not match expected type. "
                f"Expected: {cls.__name__}, Actual: {type(processor).__name__}"
            )

        return processor


class BaseFittableProcessor(BaseProcessor):
    """
    Base class for fittable data processors
    """

    @abstractmethod
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass
