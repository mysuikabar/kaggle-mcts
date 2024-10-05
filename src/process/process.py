from logging import getLogger

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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

    @staticmethod
    def _process_text_column(
        sr: pd.Series, tfidf: TfidfProcessor, fit: bool
    ) -> pd.DataFrame:
        name = sr.name
        df_result = pd.DataFrame(index=sr.index)

        # text length
        df_result[f"{name}_len"] = sr.str.len()

        # tfidf
        if fit:
            df_tfidf = tfidf.fit_transform(sr)
        else:
            df_tfidf = tfidf.transform(sr)
        df_tfidf.columns = [f"tfidf_{name}_{word}" for word in df_tfidf.columns]

        return df_result

    def _parallel_process_text_columns(
        self, df: pd.DataFrame, fit: bool
    ) -> pd.DataFrame:
        dfs = Parallel(n_jobs=-1)(
            delayed(self._process_text_column)(df[col], tfidf, fit)
            for col, tfidf in self._tfidf_container.items()
        )
        return pd.concat(dfs, axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()

        # process text columns
        logger.info("Processing text columns")
        df_text = self._parallel_process_text_columns(df_result, fit=True)
        df_result = pd.concat([df_result, df_text], axis=1)
        df_result = df_result.drop(columns=self._tfidf_container.keys())

        df_result = df_result.drop(columns=self._drop_columns, errors="ignore")
        df_result = self._cat_converter.fit_transform(df_result)

        return df_result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()

        # process text columns
        df_text = self._parallel_process_text_columns(df_result, fit=False)
        df_result = pd.concat([df_result, df_text], axis=1)
        df_result = df_result.drop(columns=self._tfidf_container.keys())

        df_result = df_result.drop(columns=self._drop_columns, errors="ignore")
        df_result = self._cat_converter.transform(df_result)

        return df_result


def postprocess(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, -1, 1)
