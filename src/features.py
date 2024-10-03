import polars as pl

from process.consts import AGENT_PATTERN, LUD_RULES_PATTERN
from process.feature import FeatureExpressions

feature_expressions_master = FeatureExpressions()


feature_expressions_master["agent_property"] = [
    pl.col("agent1").str.extract(AGENT_PATTERN, 1).alias("p1_selection"),
    pl.col("agent1")
    .str.extract(AGENT_PATTERN, 2)
    .alias("p1_exploration")
    .cast(pl.Float32),
    pl.col("agent1").str.extract(AGENT_PATTERN, 3).alias("p1_playout"),
    pl.col("agent1").str.extract(AGENT_PATTERN, 4).alias("p1_bounds"),
    pl.col("agent2").str.extract(AGENT_PATTERN, 1).alias("p2_selection"),
    pl.col("agent2")
    .str.extract(AGENT_PATTERN, 2)
    .alias("p2_exploration")
    .cast(pl.Float32),
    pl.col("agent2").str.extract(AGENT_PATTERN, 3).alias("p2_playout"),
    pl.col("agent2").str.extract(AGENT_PATTERN, 4).alias("p2_bounds"),
]


feature_expressions_master["lud_rules"] = [
    pl.col("LudRules").str.extract(LUD_RULES_PATTERN, 1).alias("LudRules_game"),
    pl.col("LudRules").str.extract(LUD_RULES_PATTERN, 2).alias("LudRules_players"),
    pl.col("LudRules").str.extract(LUD_RULES_PATTERN, 3).alias("LudRules_equipment"),
    pl.col("LudRules").str.extract(LUD_RULES_PATTERN, 4).alias("LudRules_rules"),
]
