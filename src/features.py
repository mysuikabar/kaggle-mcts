import polars as pl

from process.consts import AGENT_PATTERN, LUD_RULES_PATTERN
from process.feature import FeatureExpressions

feature_expressions_master = FeatureExpressions()


feature_expressions_master["agent_property"] = [
    pl.col("agent1").str.extract(AGENT_PATTERN, 1).alias("p1_selection"),
    pl.col("agent1").str.extract(AGENT_PATTERN, 2).alias("p1_exploration").cast(pl.Float32),
    pl.col("agent1").str.extract(AGENT_PATTERN, 3).alias("p1_playout"),
    pl.col("agent1").str.extract(AGENT_PATTERN, 4).alias("p1_bounds"),
    pl.col("agent2").str.extract(AGENT_PATTERN, 1).alias("p2_selection"),
    pl.col("agent2").str.extract(AGENT_PATTERN, 2).alias("p2_exploration").cast(pl.Float32),
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
    (pl.col("PlayoutsPerSecond") / (pl.col("MovesPerSecond") + 1e-15)).alias("Playouts/Moves"),
    (pl.col("MovesPerSecond") / (pl.col("PlayoutsPerSecond") + 1e-15)).alias("EfficiencyPerPlayout"),
    (pl.col("DurationActions") / (pl.col("DurationTurnsStdDev") + 1e-15)).alias("TurnsDurationEfficiency"),
    (pl.col("AdvantageP1") / (pl.col("Balance") + 1e-15)).alias("AdvantageBalanceRatio"),
    (pl.col("DurationActions") / (pl.col("MovesPerSecond") + 1e-15)).alias("ActionTimeEfficiency"),
    (pl.col("DurationTurnsStdDev") / (pl.col("DurationActions") + 1e-15)).alias("StandardizedTurnsEfficiency"),
    (pl.col("AdvantageP1") / (pl.col("DurationActions") + 1e-15)).alias("AdvantageTimeImpact"),
    (pl.col("DurationActions") / (pl.col("StateTreeComplexity") + 1e-15)).alias("DurationToComplexityRatio"),
    (pl.col("GameTreeComplexity") / (pl.col("StateTreeComplexity") + 1e-15)).alias("NormalizedGameTreeComplexity"),
    (pl.col("Balance") * pl.col("GameTreeComplexity")).alias("ComplexityBalanceInteraction"),
    (pl.col("StateTreeComplexity") + pl.col("GameTreeComplexity")).alias("OverallComplexity"),
    (pl.col("GameTreeComplexity") / (pl.col("PlayoutsPerSecond") + 1e-15)).alias("ComplexityPerPlayout"),
    (pl.col("DurationTurnsNotTimeouts") / (pl.col("MovesPerSecond") + 1e-15)).alias("TurnsNotTimeouts/Moves"),
    (pl.col("Timeouts") / (pl.col("DurationActions") + 1e-15)).alias("Timeouts/DurationActions"),
    (pl.col("OutcomeUniformity") / (pl.col("AdvantageP1") + 1e-15)).alias("OutcomeUniformity/AdvantageP1"),
    (pl.col("StepDecisionToEnemy") + pl.col("SlideDecisionToEnemy") + pl.col("HopDecisionMoreThanOne")).alias("ComplexDecisionRatio"),
    (pl.col("StepDecisionToEnemy") + pl.col("HopDecisionEnemyToEnemy") + pl.col("HopDecisionFriendToEnemy") + pl.col("SlideDecisionToEnemy")).alias(
        "AggressiveActionsRatio"
    ),
    # pl.col("PlayoutsPerSecond").clip(0, 25000),
    # pl.col("MovesPerSecond").clip(0, 1000000),
]


feature_expressions_master["LudRules_features"] = [
    # Basic text features
    pl.col("LudRules").str.len_chars().alias("total_length"),
    (pl.col("LudRules").str.count_matches(r"\(") + pl.col("LudRules").str.count_matches(r"\)")).alias("num_parentheses"),
    # Game phases
    pl.col("LudRules").str.contains("phases:").alias("has_phases"),
    pl.col("LudRules").str.count_matches(r'\(phase\s+"[^"]+"').alias("num_phases"),
    # Game components
    pl.col("LudRules").str.count_matches(r'\(place\s+"[^"]+"').alias("num_pieces"),
    pl.col("LudRules").str.contains("roll").alias("has_dice"),
    # Game mechanics
    pl.col("LudRules").str.contains("move").alias("has_movement"),
    pl.col("LudRules").str.contains("remove").alias("has_capture"),
    pl.col("LudRules").str.contains("Add").alias("has_addition"),
    pl.col("LudRules").str.contains("Threatened").alias("has_threats"),
    pl.col("LudRules").str.contains("is Line").alias("has_line_victory"),
    pl.col("LudRules").str.contains("is Connected").alias("has_connection_victory"),
    pl.col("LudRules").str.contains("no Pieces").alias("has_piece_count_victory"),
    # Board features
    pl.col("LudRules").str.contains(r"sites Row").alias("uses_rows"),
    pl.col("LudRules").str.contains(r"sites Col").alias("uses_columns"),
    (
        pl.col("LudRules").str.contains("NE")
        | pl.col("LudRules").str.contains("SE")
        | pl.col("LudRules").str.contains("NW")
        | pl.col("LudRules").str.contains("SW")
    ).alias("uses_diagonals"),
    pl.col("LudRules").str.contains("Orthogonal").alias("uses_orthogonal"),
    # Game state tracking
    pl.col("LudRules").str.contains("counter").alias("uses_counter"),
    pl.col("LudRules").str.contains("remember").alias("uses_memory"),
    # Control flow
    pl.col("LudRules").str.count_matches("if:").alias("num_conditionals"),
    pl.col("LudRules").str.count_matches("then").alias("num_then"),
    pl.col("LudRules").str.contains("moveAgain").alias("move_again_allowed"),
    # Operation counts
    pl.col("LudRules").str.count_matches(r"\(move\s").alias("num_move_operations"),
    pl.col("LudRules").str.count_matches(r"\(remove\s").alias("num_remove_operations"),
    pl.col("LudRules").str.count_matches(r"\(place\s").alias("num_place_operations"),
    pl.col("LudRules").str.count_matches(r"\(roll\s").alias("num_roll_operations"),
    pl.col("LudRules").str.count_matches(r"\(add\s").alias("num_add_operations"),
    # Victory conditions
    pl.col("LudRules").str.count_matches(r"\(result\s+\w+\s+Win\)").alias("num_win_conditions"),
    pl.col("LudRules").str.count_matches(r"\(result\s+\w+\s+Draw\)").alias("num_draw_conditions"),
    # Complexity indicators
    pl.col("LudRules").str.count_matches(r"\([a-zA-Z]+\s").alias("num_commands"),
    # Movement patterns
    pl.col("LudRules").str.contains(r"steps:\d+").alias("has_linear_movement"),
    pl.col("LudRules").str.contains("sites Around").alias("has_adjacent_movement"),
]
