from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from consts import REPO_ROOT

hydra_config = {
    "run": {"dir": f"{REPO_ROOT}" + "/outputs/preprocess/${now:%Y-%m-%d_%H-%M-%S}"},
    "job": {"chdir": True},
}


@dataclass
class TfidfConfig:
    text_features: list[str] = field(
        default_factory=lambda: ["EnglishRules", "LudRules_equipment", "LudRules_rules"]
    )
    max_features: int = 600


@dataclass
class Config:
    seed: int = 42
    n_splits: int = 5
    target: str = "utility_agent1"
    groups: str = "GameRulesetName"
    data_path: Path = REPO_ROOT / "data/raw/train.csv"
    feature_store_dir: Path | None = REPO_ROOT / "data/feature_store"
    tfidf: TfidfConfig = TfidfConfig()
    hydra: Any = field(default_factory=lambda: hydra_config)
