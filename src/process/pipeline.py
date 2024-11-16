import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing_extensions import Self

from .consts import USELESS_COLUMNS
from .transformers import CategoricalConverter, ColumnDropper, Tfidf


class PreprocessPipeline(TransformerMixin, BaseEstimator):
    def __init__(self, col2tfidf: dict[str, Tfidf] | None = None) -> None:
        transformers = []

        # tfidf
        if col2tfidf:
            transformers.append(
                (
                    "tfidf",
                    ColumnTransformer(
                        [
                            (f"tfidf_{col}", tfidf, col)
                            for col, tfidf in col2tfidf.items()
                        ],
                        remainder="passthrough",
                        n_jobs=len(col2tfidf),
                    ),
                )
            )

        # drop columns
        drop_columns = [
            "Id",
            "GameRulesetName",
            "agent1",
            "agent2",
            "EnglishRules",
            "LudRules",
            "LudRules_game",
            "LudRules_equipment",
            "LudRules_rules",
            "num_wins_agent1",
            "num_draws_agent1",
            "num_losses_agent1",
            "utility_agent1",
        ] + USELESS_COLUMNS
        transformers.append(("column_dropper", ColumnDropper(columns=drop_columns)))

        # categorical converter
        transformers.append(("categorical_converter", CategoricalConverter()))

        self._pipeline = Pipeline(transformers)

    def fit(self, X: pd.DataFrame, y: None = None) -> Self:
        self._pipeline.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._pipeline.transform(X)

    def save(self, filepath: str | Path) -> None:
        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        with open(filepath, "rb") as file:
            obj = pickle.load(file)

        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object type does not match expected type. "
                f"Expected: {cls.__name__}, Actual: {type(obj).__name__}"
            )

        return obj


def postprocess(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, -1, 1)
