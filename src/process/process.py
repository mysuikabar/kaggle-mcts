from logging import getLogger

import numpy as np
import pandas as pd

from .base import BaseFittableProcessor
from .consts import USELESS_COLUMNS
from .text import TfidfProcessor
from .utils import CategoricalConverter

logger = getLogger(__name__)


class PreProcessor(BaseFittableProcessor):
    def __init__(self, tfidf_max_features: int) -> None:
        self._cat_converter = CategoricalConverter()
        self._tfidf_container: dict[str, TfidfProcessor] = {
            "EnglishRules": TfidfProcessor(tfidf_max_features),
            "LudRules_equipment": TfidfProcessor(tfidf_max_features),
            "LudRules_rules": TfidfProcessor(tfidf_max_features),
        }
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

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()

        # process text columns
        for text_col, tfidf in self._tfidf_container.items():
            logger.info(f"Processing text col: {text_col}")

            # text length
            df_result[f"{text_col}_len"] = df_result[text_col].str.len()

            # tfidf
            df_tfidf = tfidf.fit_transform(df_result[text_col])
            df_tfidf.columns = [f"tfidf_{text_col}_{word}" for word in df_tfidf.columns]
            df_result = pd.concat([df_result, df_tfidf], axis=1)

            # drop original text column
            df_result = df_result.drop(columns=text_col)

        df_result = df_result.drop(columns=self._drop_columns, errors="ignore")
        df_result = self._cat_converter.fit_transform(df_result)

        return df_result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()

        # process text columns
        for text_col, tfidf in self._tfidf_container.items():
            logger.info(f"Processing text col: {text_col}")

            # text length
            df_result[f"{text_col}_len"] = df_result[text_col].str.len()

            # tfidf
            df_tfidf = tfidf.transform(df_result[text_col])
            df_tfidf.columns = [f"tfidf_{text_col}_{word}" for word in df_tfidf.columns]
            df_result = pd.concat([df_result, df_tfidf], axis=1)

            # drop original text column
            df_result = df_result.drop(columns=text_col)

        df_result = df_result.drop(columns=self._drop_columns, errors="ignore")
        df_result = self._cat_converter.transform(df_result)

        return df_result


def postprocess(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, -1, 1)
