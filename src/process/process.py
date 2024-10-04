import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from typing_extensions import Self

from .consts import USELESS_COLUMNS
from .text import TfidfProcessor
from .utils import CategoricalConverter


class Preprocessor:
    def __init__(self) -> None:
        self._cat_converter = CategoricalConverter()
        self._tfidf_container: dict[str, TfidfProcessor] = {
            "EnglishRules": TfidfProcessor(),
            "LudRules_equipment": TfidfProcessor(),
            "LudRules_rules": TfidfProcessor(),
        }

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()

        # process text columns
        for text_col, tfidf in self._tfidf_container.items():
            # text length
            df_result[f"{text_col}_len"] = df_result[text_col].str.len()

            # tfidf
            df_tfidf = tfidf.fit_transform(df_result[text_col])
            df_tfidf.columns = [f"tfidf_{text_col}_{word}" for word in df_tfidf.columns]
            df_result = pd.concat([df_result, df_tfidf], axis=1)

            # drop original text column
            df_result = df_result.drop(columns=text_col)

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
        df_result = df_result.drop(columns=self._drop_columns, errors="ignore")

        # convert object dtype to categorical
        df_result = self._cat_converter.fit_transform(df_result)

        return df_result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()

        # process text columns
        for text_col, tfidf in self._tfidf_container.items():
            # text length
            df_result[f"{text_col}_len"] = df_result[text_col].str.len()

            # tfidf
            df_tfidf = tfidf.transform(df_result[text_col])
            df_tfidf.columns = [f"tfidf_{text_col}_{word}" for word in df_tfidf.columns]
            df_result = pd.concat([df_result, df_tfidf], axis=1)

            # drop original text column
            df_result = df_result.drop(columns=text_col)

        # drop columns
        df_result = df_result.drop(columns=self._drop_columns, errors="ignore")

        # convert object dtype to categorical
        df_result = self._cat_converter.transform(df_result)

        return df_result

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
