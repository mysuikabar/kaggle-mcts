import numpy as np
import pandas as pd
import polars as pl


def _drop_constant_columns(df: pl.DataFrame) -> pl.DataFrame:
    constant_columns = np.array(df.columns)[
        df.select(pl.all().n_unique() == 1).to_numpy().ravel()
    ]
    drop_columns = list(constant_columns) + ["Id"]

    return df.drop(drop_columns)


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
        pl.col("agent1").str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 4).alias("p1_bounds"),
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
        pl.col("agent2").str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 4).alias("p2_bounds"),
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


def _convert_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_mapping = {
        feature: pd.CategoricalDtype(categories=list(set(df[feature])))
        for feature in df.columns[df.dtypes == object]  # noqa
    }
    return df.astype(cat_mapping)


def preprocess(
    df: pl.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None]:
    df_feature = (
        df.pipe(_drop_constant_columns)
        .pipe(_create_features)
        .to_pandas()
        .pipe(_convert_to_categorical)
    )

    if "utility_agent1" in df.columns:
        target = df.select("utility_agent1").to_numpy().ravel()
        groups = df.select("GameRulesetName").to_numpy()
    else:
        target, groups = None, None

    return df_feature, target, groups
