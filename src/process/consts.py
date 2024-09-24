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
