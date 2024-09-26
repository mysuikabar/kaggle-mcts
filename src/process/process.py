import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from typing_extensions import Self

from .consts import USELESS_COLUMNS
from .feature import FeatureProcessor
from .utils import CategoricalConverter


class Preprocessor:
    def __init__(self, feature_processor: FeatureProcessor) -> None:
        self._feature_processor = feature_processor
        self._cat_converter = CategoricalConverter()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pl.DataFrame(df)

        # feature engineering
        df_result = self._feature_processor.run(df)

        # drop columns
        self._drop_columns = [
            "Id",
            "GameRulesetName",
            "EnglishRules",
            "LudRules",
            "num_wins_agent1",
            "num_draws_agent1",
            "num_losses_agent1",
            "utility_agent1",
        ] + USELESS_COLUMNS
        df_result = df_result.drop(self._drop_columns, strict=False)

        # convert dtype to categorical
        df_result = self._cat_converter.fit_transform(df_result.to_pandas())

        return df_result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pl.DataFrame(df)

        df_result = self._feature_processor.run(df)
        df_result = df_result.drop(self._drop_columns, strict=False)
        df_result = self._cat_converter.transform(df_result.to_pandas())

        return df_result

    def disable_feature_store(self) -> None:
        self._feature_processor.disable_feature_store()

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


def postprocess(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, -1, 1)
