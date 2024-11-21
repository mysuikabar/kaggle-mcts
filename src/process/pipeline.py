import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from typing_extensions import Self

from .consts import USELESS_COLUMNS
from .transformers import (
    CategoricalConverter,
    ColumnDropper,
    ColumnSelector,
    IdentityTransformer,
    Tfidf,
)


class PreprocessPipeline(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        col2tfidf: dict[str, Tfidf] | None = None,
        use_columns: list[str] | None = None,
    ) -> None:
        self.col2tfidf = col2tfidf
        self.use_columns = use_columns

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

        # create pipeline
        transformers = []

        if col2tfidf:
            tfidfs = ColumnTransformer([(f"tfidf_{col}", tfidf, col) for col, tfidf in col2tfidf.items()], remainder="drop", n_jobs=len(col2tfidf))
            transformers += [("add_tfidf", FeatureUnion([("tfidfs", tfidfs), ("identity", IdentityTransformer())]))]

        transformers += [
            ("drop_columns", ColumnDropper(drop_columns)),
            ("categorical", CategoricalConverter()),
        ]

        if use_columns:
            transformers += [("select_columns", ColumnSelector(use_columns))]

        self._pipeline = Pipeline(transformers).set_output(transform="pandas")

    def fit(self, X: pd.DataFrame, y: None = None) -> Self:
        self._pipeline.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._pipeline.transform(X)


def postprocess(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred * 1.1, -0.985, 0.985)  # np.clip(pred, -1, 1)
