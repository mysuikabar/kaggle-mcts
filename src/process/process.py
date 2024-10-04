import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from typing_extensions import Self

from .consts import USELESS_COLUMNS
from .feature import FeatureProcessor
from .text import TfidfProcessor
from .utils import CategoricalConverter


class Preprocessor:
    def __init__(self, feature_processor: FeatureProcessor) -> None:
        self._feature_processor = feature_processor
        self._cat_converter = CategoricalConverter()
        self._tfidf_container: dict[str, TfidfProcessor] = {
            "EnglishRules": TfidfProcessor(),
            "LudRules_equipment": TfidfProcessor(),
            "LudRules_rules": TfidfProcessor(),
        }

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pl.DataFrame(df)

        # feature engineering
        df_result = self._feature_processor.run(df)

        # process text columns
        for text_col, tfidf in self._tfidf_container.items():
            df_tfidf = tfidf.fit_transform(df_result[text_col].to_pandas())
            df_tfidf.columns = [f"tfidf_{text_col}_{word}" for word in df_tfidf.columns]
            df_result = (
                df_result.with_columns(
                    pl.col(text_col).str.len_chars().alias(f"{text_col}_len_chars")
                )
                .hstack(pl.DataFrame(df_tfidf))
                .drop(text_col, strict=False)
            )

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

        for text_col, tfidf in self._tfidf_container.items():
            df_tfidf = tfidf.transform(df_result[text_col].to_pandas())
            df_tfidf.columns = [f"tfidf_{text_col}_{word}" for word in df_tfidf.columns]
            df_result = (
                df_result.with_columns(
                    pl.col(text_col).str.len_chars().alias(f"{text_col}_len_chars")
                )
                .hstack(pl.DataFrame(df_tfidf))
                .drop(text_col, strict=False)
            )

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
