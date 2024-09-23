from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from consts import REPO_ROOT
from ml.model.base import BaseConfig

from .model.gbdt import lightgbm_config


@dataclass
class ModelConfig:
    type: str = "lightgbm"
    config: BaseConfig = lightgbm_config
    output_dir: Path = Path("models")


@dataclass
class PreprocessConfig:
    use_features: list[str] = field(default_factory=lambda: ["agent_property"])
    feature_store_dir: Path | None = REPO_ROOT / "data" / "feature_store"


@dataclass
class WandbConfig:
    project: str = "kaggle-mcts"
    name: str = "run_name"
    enable: bool = True


hydra_config = {
    "run": {"dir": f"{REPO_ROOT}/outputs/" + WandbConfig.name},
    "sweep": {
        "dir": f"{REPO_ROOT}/outputs/" + WandbConfig.name,
        "subdir": "${hydra.job.override_dirname}",
    },
    "job": {"chdir": True},
}


@dataclass
class Config:
    seed: int = 42
    data_path: Path = REPO_ROOT / "data" / "raw" / "train.csv"
    preprocess: PreprocessConfig = PreprocessConfig()
    model: ModelConfig = ModelConfig()
    wandb: WandbConfig = WandbConfig()
    hydra: Any = field(default_factory=lambda: hydra_config)
