from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from consts import REPO_ROOT
from ml.model.base import BaseConfig

from .model.gbdt import lightgbm_config

hydra_config = {
    "run": {"dir": f"{REPO_ROOT}/outputs/" + "${now:%Y-%m-%d_%H-%M-%S}"},
    "sweep": {
        "dir": f"{REPO_ROOT}/outputs/" + "${now:%Y-%m-%d_%H-%M-%S}",
        "subdir": "${hydra.job.override_dirname}",
    },
    "job": {"chdir": True},
}


@dataclass
class ModelConfig:
    type: str = "lightgbm"
    config: BaseConfig = lightgbm_config
    output_dir: Path = Path("models")


@dataclass
class PreprocessConfig:
    use_features: list[str] = field(default_factory=lambda: ["agent_property"])
    feature_store_dir: Path = REPO_ROOT / "data" / "feature_store"


@dataclass
class WandbConfig:
    enable: bool = True
    project: str = "kaggle-mcts"


@dataclass
class Config:
    seed: int = 42
    data_path: Path = REPO_ROOT / "data" / "raw" / "train_mini.csv"
    preprocess: PreprocessConfig = PreprocessConfig()
    model: ModelConfig = ModelConfig()
    wandb: WandbConfig = WandbConfig()
    hydra: Any = field(default_factory=lambda: hydra_config)
