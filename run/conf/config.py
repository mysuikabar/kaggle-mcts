from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from consts import REPO_ROOT

from .model import (  # noqa: F401
    CatBoostConfig,
    LightGBMConfig,
    NNConfig,
    XGBoostConfig,
)


@dataclass
class FeatureConfig:
    use_features: list[str] = field(
        default_factory=lambda: ["agent_property", "lud_rules", "baseline_features"]
    )
    store_dir: Path | None = None  # REPO_ROOT / "data/feature_store"


@dataclass
class ModelConfig:
    type: str = "catboost"
    config: CatBoostConfig = CatBoostConfig()


@dataclass
class WandbConfig:
    project: str = "kaggle-mcts"
    name: str = "run_name"
    notes: str | None = "description"
    enable: bool = False


hydra_config = {
    "run": {"dir": f"{REPO_ROOT}/outputs/{WandbConfig.name}"},
    "sweep": {
        "dir": f"{REPO_ROOT}/outputs/{WandbConfig.name}",
        "subdir": "${hydra.job.override_dirname}",
    },
    "job": {"chdir": True},
    "verbose": True,
}


@dataclass
class Config:
    seed: int = 42
    target: str = "utility_agent1"
    data_path: Path = REPO_ROOT / "data/raw/train.csv"
    preprocess_dir: Path = REPO_ROOT / "outputs/preprocess/mcts-01-preprocess"
    importance_dir: Path | None = REPO_ROOT / "outputs/mcts-11-cbt"
    num_features: int = 500
    feature: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    wandb: WandbConfig = WandbConfig()
    hydra: Any = field(default_factory=lambda: hydra_config)
