from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from consts import REPO_ROOT
from ml.model.base import BaseConfig

from .model.gbdt import lightgbm_config


@dataclass
class FeatureConfig:
    use_features: list[str] = field(
        default_factory=lambda: ["agent_property", "lud_rules"]
    )
    feature_store_dir: Path | None = None


@dataclass
class ModelConfig:
    type: str = "lightgbm"
    config: BaseConfig = lightgbm_config


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
    n_splits: int = 5
    target: str = "utility_agent1"
    groups: str = "GameRulesetName"
    data_path: Path = REPO_ROOT / "data/raw/train.csv"
    feature: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    wandb: WandbConfig = WandbConfig()
    hydra: Any = field(default_factory=lambda: hydra_config)
