from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from consts import REPO_ROOT
from ml.model.base import BaseConfig
from ml.model.gbdt import LightGBMConfig

hydra_config = {
    "run": {"dir": f"{REPO_ROOT}/outputs/" + "${now:%Y-%m-%d_%H-%M-%S}"},
    "sweep": {
        "dir": f"{REPO_ROOT}/outputs/" + "${now:%Y-%m-%d_%H-%M-%S}",
        "subdir": "${hydra.job.override_dirname}",
    },
    "job": {"chdir": True},
}


model_confi = LightGBMConfig(
    objective="regression",
    metric="rmse",
    boosting_type="gbdt",
    num_leaves=30,
    learning_rate=0.05,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    num_boost_round=1000,
    early_stopping_rounds=10,
)


@dataclass
class Config:
    data_path: Path = REPO_ROOT / "data" / "raw" / "train_mini.csv"
    model_type: str = "lightgbm"
    model_config: BaseConfig = LightGBMConfig()
    model_output_dir: Path = Path("models")
    hydra: Any = field(default_factory=lambda: hydra_config)
