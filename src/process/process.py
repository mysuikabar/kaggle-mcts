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
        df: pd.DataFrame, col: str, tfidf: TfidfProcessor, fit: bool
    ) -> tuple[pd.DataFrame, str, TfidfProcessor]:
        df_result = pd.DataFrame(index=df.index)

        # text length
        df_result[f"{col}_len"] = df[col].str.len()

        # tfidf
        if fit:
            df_tfidf = tfidf.fit_transform(df[col])
        else:
            df_tfidf = tfidf.transform(df[col])
        df_tfidf.columns = [f"tfidf_{col}_{word}" for word in df_tfidf.columns]

        df_result = pd.concat([df_result, df_tfidf], axis=1)

        return df_result, col, tfidf

    @staticmethod
    def _parallel_process_text_columns(
        df: pd.DataFrame, col2tfidf: dict[str, TfidfProcessor], fit: bool
    ) -> tuple[pd.DataFrame, dict[str, TfidfProcessor]]:
        df_result = pd.DataFrame(index=df.index)

        results = Parallel(n_jobs=-1)(
            delayed(PreProcessor._process_text_column)(df, col, tfidf, fit)
            for col, tfidf in col2tfidf.items()
        )

        for df_col, col, tfidf in results:
            df_result = pd.concat([df_result, df_col], axis=1)
            col2tfidf[col] = tfidf

        return df_result, col2tfidf

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()

        # process text columns
        logger.info("Processing text columns")
        df_text, self._tfidf_container = self._parallel_process_text_columns(
            df_result, self._tfidf_container, fit=True
        )
        df_result = pd.concat([df_result, df_text], axis=1)
        df_result = df_result.drop(columns=self._tfidf_container.keys())

        df_result = df_result.drop(columns=self._drop_columns, errors="ignore")
        df_result = self._cat_converter.fit_transform(df_result)

        return df_result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()

        # process text columns
        df_text, _ = self._parallel_process_text_columns(
            df_result, self._tfidf_container, fit=False
        )
        df_result = pd.concat([df_result, df_text], axis=1)
        df_result = df_result.drop(columns=self._tfidf_container.keys())

        df_result = df_result.drop(columns=self._drop_columns, errors="ignore")
        df_result = self._cat_converter.transform(df_result)

        return df_result


def postprocess(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, -1, 1)
