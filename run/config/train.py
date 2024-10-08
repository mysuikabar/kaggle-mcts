from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from consts import REPO_ROOT
from ml.model.base import BaseConfig

from .model.gbdt import xgboost_config


@dataclass
class FeatureConfig:
    use_features: list[str] = field(
        default_factory=lambda: ["agent_property", "lud_rules"]
    )
    feature_store_dir: Path | None = None  # REPO_ROOT / "data/feature_store"


@dataclass
class ModelConfig:
    type: str = "xgboost"
    config: BaseConfig = xgboost_config


@dataclass
class WandbConfig:
    project: str = "kaggle-mcts"
    name: str = "run_name"
    enable: bool = False


hydra_config = {
    "run": {"dir": f"{REPO_ROOT}/outputs/{WandbConfig.name}"},
    "sweep": {
        "dir": f"{REPO_ROOT}/outputs/{WandbConfig.name}",
        "subdir": "${hydra.job.override_dirname}",
    },
    "job": {"chdir": True},
}


@dataclass
class Config:
    seed: int = 42
    target: str = "utility_agent1"
    data_path: Path = REPO_ROOT / "data/raw/train.csv"
    preprocess_dir: Path = REPO_ROOT / "outputs/preprocess/mcts-01-preprocess"
    feature: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    wandb: WandbConfig = WandbConfig()
    hydra: Any = field(default_factory=lambda: hydra_config)