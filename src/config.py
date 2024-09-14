from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from consts import REPO_ROOT

hydra_config = {
    "run": {"dir": f"{REPO_ROOT}/outputs/" + "${now:%Y-%m-%d_%H-%M-%S}"},
    "sweep": {
        "dir": f"{REPO_ROOT}/outputs/" + "${now:%Y-%m-%d_%H-%M-%S}",
        "subdir": "${hydra.job.override_dirname}",
    },
    "job": {"chdir": True},
}


@dataclass
class ModelParams:
    objective: str = "regression"
    metric: str = "rmse"
    boosting_type: str = "gbdt"
    num_leaves: int = 30
    learning_rate: float = 0.05
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    num_boost_round: int = 1000
    early_stopping_rounds: int = 10


@dataclass
class Config:
    data_path: Path = REPO_ROOT / "data" / "train_mini.csv"
    model_type: str = "lightgbm"
    model_params: ModelParams = ModelParams()
    hydra: Any = field(default_factory=lambda: hydra_config)
