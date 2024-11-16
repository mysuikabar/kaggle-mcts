from dataclasses import dataclass, field


@dataclass
class LightGBMConfig:
    objective: str = "regression"
    metric: str = "rmse"
    boosting_type: str = "gbdt"
    num_leaves: int = 30
    learning_rate: float = 0.05
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    num_boost_round: int = 10000
    early_stopping_rounds: int = 50
    max_bin: int = 255
    device: str = "gpu"


@dataclass
class XGBoostConfig:
    objective: str = "reg:squarederror"
    eval_metric: str = "rmse"
    booster: str = "gbtree"
    learning_rate: float = 0.05
    min_split_loss: float = 0
    max_depth: int = 6
    min_child_weight: float = 1
    subsample: float = 0.8
    reg_lambda: float = 1
    reg_alpha: float = 1
    colsample_bytree: float = 1
    colsample_bylevel: float = 1
    num_boost_round: int = 10000
    early_stopping_rounds: int = 50
    device: str = "gpu"


@dataclass
class CatBoostConfig:
    loss_function: str = "RMSE"
    eval_metric: str = "RMSE"
    learning_rate: float = 0.05
    depth: int = 12
    l2_leaf_reg: float = 1
    random_strength: float = 1
    bagging_temperature: float = 0.5
    od_type: str = "Iter"
    od_wait: int = 10
    iterations: int = 10000
    early_stopping_rounds: int = 50
    task_type: str = "GPU"


@dataclass
class NNConfig:
    categorical_feature_dims: dict[str, int] = field(
        default_factory=lambda: {
            "p1_selection": 4,
            "p1_playout": 3,
            "p1_bounds": 2,
            "p2_selection": 4,
            "p2_playout": 3,
            "p2_bounds": 2,
            "LudRules_players": 9,
        }
    )
    embedding_dim: int = 4
    hidden_dims: list[int] = field(default_factory=lambda: [1024, 512, 256])
    dropout_rate: float = 0.1
    learning_rate: float = 1e-4
    scheduler_patience: int = 5
    max_epochs: int = 1000
    early_stopping_patience: int = 10
    batch_size: int = 512
