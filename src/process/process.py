import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from typing_extensions import Self

from .utils import CategoricalConverter, constant_columns


class Preprocessor:
    def __init__(self) -> None:
        self._cat_converter = CategoricalConverter()

    @staticmethod
    def _create_features(df: pl.DataFrame) -> pl.DataFrame:
        df_feature = df.with_columns(
            pl.col("agent1")
            .str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 1)
            .alias("p1_selection"),
            pl.col("agent1")
            .str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 2)
            .alias("p1_exploration")
            .cast(pl.Float32),
            pl.col("agent1")
            .str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 3)
            .alias("p1_playout"),
            pl.col("agent1")
            .str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 4)
            .alias("p1_bounds"),
            pl.col("agent2")
            .str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 1)
            .alias("p2_selection"),
            pl.col("agent2")
            .str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 2)
            .alias("p2_exploration")
            .cast(pl.Float32),
            pl.col("agent2")
            .str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 3)
            .alias("p2_playout"),
            pl.col("agent2")
            .str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 4)
            .alias("p2_bounds"),
        ).drop(
            [
                "GameRulesetName",
                "EnglishRules",
                "LudRules",
                "num_wins_agent1",
                "num_draws_agent1",
                "num_losses_agent1",
                "utility_agent1",
            ],
            strict=False,
        )

        return df_feature

    def fit_transform(self, df: pl.DataFrame) -> pd.DataFrame:
        # get group label
        self._group_label = df.select("GameRulesetName").to_numpy()

        # feature engineering
        self._drop_columns = ["Id"] + constant_columns(df)
        df_feature = df.drop(self._drop_columns).pipe(self._create_features).to_pandas()

        # convert dtype to categorical
        self._cat_converter.fit(df_feature)
        df_feature = self._cat_converter.fit_transform(df_feature)

        return df_feature

    def transform(self, df: pl.DataFrame) -> pd.DataFrame:
        df_feature = df.drop(self._drop_columns).pipe(self._create_features).to_pandas()
        df_feature = self._cat_converter.transform(df_feature)

        return df_feature

    @property
    def get_group_label(self) -> np.ndarray:
        return self._group_label.copy()

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
