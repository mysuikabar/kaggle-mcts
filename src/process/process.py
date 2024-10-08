from logging import getLogger

import numpy as np
import pandas as pd

from .base import BaseFittableProcessor
from .consts import USELESS_COLUMNS
from .text import TfidfProcessor, parallel_transform_tfidf
from .utils import CategoricalConverter

logger = getLogger(__name__)


class PreProcessor(BaseFittableProcessor):
    def __init__(self, col2tfidf: dict[str, TfidfProcessor]) -> None:
        self._cat_converter = CategoricalConverter()
        self._col2tfidf = col2tfidf
        self._drop_columns = [
            "Id",
            "GameRulesetName",
            "agent1",
            "agent2",
            "EnglishRules",
            "LudRules",
            "LudRules_game",
            "num_wins_agent1",
            "num_draws_agent1",
            "num_losses_agent1",
            "utility_agent1",
        ] + USELESS_COLUMNS

    def _transform(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df_result = df.copy()

        # tfidf
        logger.info("Transforming tf-idf")
        df_tfidf = parallel_transform_tfidf(df, self._col2tfidf)
        df_result = pd.concat([df_result, df_tfidf], axis=1)
        df_result = df_result.drop(columns=self._col2tfidf.keys())

        # drop columns
        df_result = df_result.drop(columns=self._drop_columns, errors="ignore")

        # convert categorical columns
        logger.info("Converting categorical columns")
        if fit:
            df_result = self._cat_converter.fit_transform(df_result)
        else:
            df_result = self._cat_converter.transform(df_result)

        return df_result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._transform(df, fit=False)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._transform(df, fit=True)


def postprocess(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, -1, 1)
