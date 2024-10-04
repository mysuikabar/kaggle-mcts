# text columns pattern
AGENT_PATTERN = r"MCTS-(.*)-(.*)-(.*)-(.*)"
LUD_RULES_PATTERN = r"^\(game\s+\"(.*?)\"\s+\(players\s+(.*?)\)\s+\(equipment\s+(.*?)\)\s+\(rules\s+(.*?)\)\s+\)$"

# stopwords
STOP_WORDS = [
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "ain",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "couldn",
    "couldn't",
    "d",
    "did",
    "didn",
    "didn't",
    "do",
    "does",
    "doesn",
    "doesn't",
    "doing",
    "don",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn",
    "hadn't",
    "has",
    "hasn",
    "hasn't",
    "have",
    "haven",
    "haven't",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "just",
    "ll",
    "m",
    "ma",
    "me",
    "mightn",
    "mightn't",
    "more",
    "most",
    "mustn",
    "mustn't",
    "my",
    "myself",
    "needn",
    "needn't",
    "no",
    "nor",
    "not",
    "now",
    "o",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "re",
    "s",
    "same",
    "shan",
    "shan't",
    "she",
    "she's",
    "should",
    "should've",
    "shouldn",
    "shouldn't",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "that'll",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "ve",
    "very",
    "was",
    "wasn",
    "wasn't",
    "we",
    "were",
    "weren",
    "weren't",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
    "y",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
]


# useless columns
_null_columns = [
    "Behaviour",
    "StateRepetition",
    "Duration",
    "Complexity",
    "BoardCoverage",
    "GameOutcome",
    "StateEvaluation",
    "Clarity",
    "Decisiveness",
    "Drama",
    "MoveEvaluation",
    "StateEvaluationDifference",
    "BoardSitesOccupied",
    "BranchingFactor",
    "DecisionFactor",
    "MoveDistance",
    "PieceNumber",
    "ScoreDifference",
]

_constant_columns = [
    "Properties",
    "Format",
    "Time",
    "Discrete",
    "Realtime",
    "Turns",
    "Alternating",
    "Simultaneous",
    "HiddenInformation",
    "Match",
    "AsymmetricRules",
    "AsymmetricPlayRules",
    "AsymmetricEndRules",
    "AsymmetricSetup",
    "Players",
    "NumPlayers",
    "Simulation",
    "Solitaire",
    "TwoPlayer",
    "Multiplayer",
    "Coalition",
    "Puzzle",
    "DeductionPuzzle",
    "PlanningPuzzle",
    "Equipment",
    "Container",
    "Board",
    "PrismShape",
    "ParallelogramShape",
    "RectanglePyramidalShape",
    "TargetShape",
    "BrickTiling",
    "CelticTiling",
    "QuadHexTiling",
    "Hints",
    "PlayableSites",
    "Component",
    "DiceD3",
    "BiasedDice",
    "Card",
    "Domino",
    "Rules",
    "SituationalTurnKo",
    "SituationalSuperko",
    "InitialAmount",
    "InitialPot",
    "Play",
    "BetDecision",
    "BetDecisionFrequency",
    "VoteDecisionFrequency",
    "ChooseTrumpSuitDecision",
    "ChooseTrumpSuitDecisionFrequency",
    "LeapDecisionToFriend",
    "LeapDecisionToFriendFrequency",
    "HopDecisionEnemyToFriend",
    "HopDecisionEnemyToFriendFrequency",
    "HopDecisionFriendToFriend",
    "FromToDecisionWithinBoard",
    "FromToDecisionBetweenContainers",
    "BetEffect",
    "BetEffectFrequency",
    "VoteEffectFrequency",
    "SwapPlayersEffectFrequency",
    "TakeControl",
    "TakeControlFrequency",
    "PassEffectFrequency",
    "SetCost",
    "SetCostFrequency",
    "SetPhase",
    "SetPhaseFrequency",
    "SetTrumpSuit",
    "SetTrumpSuitFrequency",
    "StepEffectFrequency",
    "SlideEffectFrequency",
    "LeapEffectFrequency",
    "HopEffectFrequency",
    "FromToEffectFrequency",
    "SwapPiecesEffect",
    "SwapPiecesEffectFrequency",
    "ShootEffect",
    "ShootEffectFrequency",
    "MaxCapture",
    "OffDiagonalDirection",
    "Information",
    "HidePieceType",
    "HidePieceOwner",
    "HidePieceCount",
    "HidePieceRotation",
    "HidePieceValue",
    "HidePieceState",
    "InvisiblePiece",
    "End",
    "LineDrawFrequency",
    "ConnectionDraw",
    "ConnectionDrawFrequency",
    "GroupLossFrequency",
    "GroupDrawFrequency",
    "LoopLossFrequency",
    "LoopDraw",
    "LoopDrawFrequency",
    "PatternLoss",
    "PatternLossFrequency",
    "PatternDraw",
    "PatternDrawFrequency",
    "PathExtentEndFrequency",
    "PathExtentWinFrequency",
    "PathExtentLossFrequency",
    "PathExtentDraw",
    "PathExtentDrawFrequency",
    "TerritoryLoss",
    "TerritoryLossFrequency",
    "TerritoryDraw",
    "TerritoryDrawFrequency",
    "CheckmateLoss",
    "CheckmateLossFrequency",
    "CheckmateDraw",
    "CheckmateDrawFrequency",
    "NoTargetPieceLoss",
    "NoTargetPieceLossFrequency",
    "NoTargetPieceDraw",
    "NoTargetPieceDrawFrequency",
    "NoOwnPiecesDraw",
    "NoOwnPiecesDrawFrequency",
    "FillLoss",
    "FillLossFrequency",
    "FillDraw",
    "FillDrawFrequency",
    "ScoringDrawFrequency",
    "NoProgressWin",
    "NoProgressWinFrequency",
    "NoProgressLoss",
    "NoProgressLossFrequency",
    "SolvedEnd",
    "PositionalRepetition",
    "SituationalRepetition",
    "Narrowness",
    "Variance",
    "DecisivenessMoves",
    "DecisivenessThreshold",
    "LeadChange",
    "Stability",
    "DramaAverage",
    "DramaMedian",
    "DramaMaximum",
    "DramaMinimum",
    "DramaVariance",
    "DramaChangeAverage",
    "DramaChangeSign",
    "DramaChangeLineBestFit",
    "DramaChangeNumTimes",
    "DramaMaxIncrease",
    "DramaMaxDecrease",
    "MoveEvaluationAverage",
    "MoveEvaluationMedian",
    "MoveEvaluationMaximum",
    "MoveEvaluationMinimum",
    "MoveEvaluationVariance",
    "MoveEvaluationChangeAverage",
    "MoveEvaluationChangeSign",
    "MoveEvaluationChangeLineBestFit",
    "MoveEvaluationChangeNumTimes",
    "MoveEvaluationMaxIncrease",
    "MoveEvaluationMaxDecrease",
    "StateEvaluationDifferenceAverage",
    "StateEvaluationDifferenceMedian",
    "StateEvaluationDifferenceMaximum",
    "StateEvaluationDifferenceMinimum",
    "StateEvaluationDifferenceVariance",
    "StateEvaluationDifferenceChangeAverage",
    "StateEvaluationDifferenceChangeSign",
    "StateEvaluationDifferenceChangeLineBestFit",
    "StateEvaluationDifferenceChangeNumTimes",
    "StateEvaluationDifferenceMaxIncrease",
    "StateEvaluationDifferenceMaxDecrease",
    "BoardSitesOccupiedMinimum",
    "BranchingFactorMinimum",
    "DecisionFactorMinimum",
    "MoveDistanceMinimum",
    "PieceNumberMinimum",
    "ScoreDifferenceMinimum",
    "ScoreDifferenceChangeNumTimes",
    "Roots",
    "Cosine",
    "Sine",
    "Tangent",
    "Exponential",
    "Logarithm",
    "ExclusiveDisjunction",
    "Float",
    "HandComponent",
    "SetHidden",
    "SetInvisible",
    "SetHiddenCount",
    "SetHiddenRotation",
    "SetHiddenState",
    "SetHiddenValue",
    "SetHiddenWhat",
    "SetHiddenWho",
]


_repeated_columns = [
    "AsymmetricForces",
    "AsymmetricPiecesType",
    "BackwardRightDirection",
    "CircleTiling",
    "ForwardRightDirection",
    "LeftwardDirection",
    "LeftwardsDirection",
    "LoopEnd",
    "LoopLoss",
    "LoopWinFrequency",
    "MancalaStyle",
    "NoProgressDrawFrequency",
    "NumPerimeterSites",
    "PathExtent",
    "PathExtentEnd",
    "PathExtentLoss",
    "PathExtentWin",
    "PatternWin",
    "PatternWinFrequency",
    "PieceDirection",
    "Roll",
    "SetRotation",
    "SetRotationFrequency",
    "ShibumiStyle",
    "Sow",
    "SowOriginFirst",
    "SpiralTiling",
    "StackState",
    "SwapOption",
    "Team",
    "TerritoryEnd",
    "TerritoryWin",
    "TerritoryWinFrequency",
]


USELESS_COLUMNS = list(set(_null_columns + _constant_columns + _repeated_columns))
