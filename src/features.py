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


# ref: https://www.kaggle.com/code/yunsuxiaozi/mcts-starter
feature_expressions_master["baseline_features"] = [
    (pl.col("NumRows") * pl.col("NumColumns")).alias("area"),
    (pl.col("NumColumns").eq(pl.col("NumRows"))).cast(pl.Int8).alias("row_equal_col"),
    (pl.col("PlayoutsPerSecond") / (pl.col("MovesPerSecond") + 1e-15)).alias(
        "Playouts/Moves"
    ),
    (pl.col("MovesPerSecond") / (pl.col("PlayoutsPerSecond") + 1e-15)).alias(
        "EfficiencyPerPlayout"
    ),
    (pl.col("DurationActions") / (pl.col("DurationTurnsStdDev") + 1e-15)).alias(
        "TurnsDurationEfficiency"
    ),
    (pl.col("AdvantageP1") / (pl.col("Balance") + 1e-15)).alias(
        "AdvantageBalanceRatio"
    ),
    (pl.col("DurationActions") / (pl.col("MovesPerSecond") + 1e-15)).alias(
        "ActionTimeEfficiency"
    ),
    (pl.col("DurationTurnsStdDev") / (pl.col("DurationActions") + 1e-15)).alias(
        "StandardizedTurnsEfficiency"
    ),
    (pl.col("AdvantageP1") / (pl.col("DurationActions") + 1e-15)).alias(
        "AdvantageTimeImpact"
    ),
    (pl.col("DurationActions") / (pl.col("StateTreeComplexity") + 1e-15)).alias(
        "DurationToComplexityRatio"
    ),
    (pl.col("GameTreeComplexity") / (pl.col("StateTreeComplexity") + 1e-15)).alias(
        "NormalizedGameTreeComplexity"
    ),
    (pl.col("Balance") * pl.col("GameTreeComplexity")).alias(
        "ComplexityBalanceInteraction"
    ),
    (pl.col("StateTreeComplexity") + pl.col("GameTreeComplexity")).alias(
        "OverallComplexity"
    ),
    (pl.col("GameTreeComplexity") / (pl.col("PlayoutsPerSecond") + 1e-15)).alias(
        "ComplexityPerPlayout"
    ),
    (pl.col("DurationTurnsNotTimeouts") / (pl.col("MovesPerSecond") + 1e-15)).alias(
        "TurnsNotTimeouts/Moves"
    ),
    (pl.col("Timeouts") / (pl.col("DurationActions") + 1e-15)).alias(
        "Timeouts/DurationActions"
    ),
    (pl.col("OutcomeUniformity") / (pl.col("AdvantageP1") + 1e-15)).alias(
        "OutcomeUniformity/AdvantageP1"
    ),
    (
        pl.col("StepDecisionToEnemy")
        + pl.col("SlideDecisionToEnemy")
        + pl.col("HopDecisionMoreThanOne")
    ).alias("ComplexDecisionRatio"),
    (
        pl.col("StepDecisionToEnemy")
        + pl.col("HopDecisionEnemyToEnemy")
        + pl.col("HopDecisionFriendToEnemy")
        + pl.col("SlideDecisionToEnemy")
    ).alias("AggressiveActionsRatio"),
    # pl.col("PlayoutsPerSecond").clip(0, 25000),
    # pl.col("MovesPerSecond").clip(0, 1000000),
]
