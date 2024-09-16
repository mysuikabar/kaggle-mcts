import polars as pl

from .manager import FeatureExpressions

feature_expressions_master = FeatureExpressions()

feature_expressions_master["agent_property"] = [
    pl.col("agent1").str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 1).alias("p1_selection"),
    pl.col("agent1")
    .str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 2)
    .alias("p1_exploration")
    .cast(pl.Float32),
    pl.col("agent1").str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 3).alias("p1_playout"),
    pl.col("agent1").str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 4).alias("p1_bounds"),
    pl.col("agent2").str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 1).alias("p2_selection"),
    pl.col("agent2")
    .str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 2)
    .alias("p2_exploration")
    .cast(pl.Float32),
    pl.col("agent2").str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 3).alias("p2_playout"),
    pl.col("agent2").str.extract(r"MCTS-(.*)-(.*)-(.*)-(.*)", 4).alias("p2_bounds"),
]
