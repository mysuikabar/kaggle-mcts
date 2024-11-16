import pickle
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold

from consts import REPO_ROOT
from features import feature_expressions_master
from process.feature import FeatureProcessor
from process.transformers import Tfidf
from utils.seed import seed_everything

hydra_config = {
    "run": {"dir": f"{REPO_ROOT}" + "/outputs/preprocess/${now:%Y-%m-%d_%H-%M-%S}"},
    "job": {"chdir": True},
}


@dataclass
class TfidfConfig:
    text_features: list[str] = field(default_factory=lambda: ["EnglishRules", "LudRules_equipment", "LudRules_rules"])
    max_features: int = 600


@dataclass
class Config:
    seed: int = 42
    n_splits: int = 5
    target: str = "utility_agent1"
    groups: str = "GameRulesetName"
    data_path: Path = REPO_ROOT / "data/raw/train.csv"
    tfidf: TfidfConfig = TfidfConfig()
    hydra: Any = field(default_factory=lambda: hydra_config)


logger = getLogger(__name__)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(config: Config) -> None:
    seed_everything(config.seed)

    # load data
    df = pd.read_csv(config.data_path)
    X = df.drop(columns=[config.target])
    groups = df[config.groups].values
    logger.info(f"Raw data shape: {df.shape}")

    # feature engineering
    feature_processor = FeatureProcessor(feature_expressions_master)
    X = feature_processor.transform(X)
    logger.info(f"Feature added data shape: {X.shape}")

    # process by fold
    fold_assignments = np.zeros(len(X), dtype=int)

    kf = GroupKFold(config.n_splits)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(X, groups=groups), start=1):
        logger.info(f"Processing fold {fold}")
        X_tr = X.iloc[idx_tr]

        # tf-idf
        logger.info("Fitting tf-idf")
        transformer = ColumnTransformer(
            [(feature, Tfidf(max_features=config.tfidf.max_features), feature) for feature in config.tfidf.text_features],
            remainder="passthrough",
            n_jobs=len(config.tfidf.text_features),
        )
        transformer.fit(X_tr)

        # save tf-idf instances as pickle
        Path(f"fold_{fold}").mkdir(exist_ok=True)
        for i, feature in enumerate(config.tfidf.text_features):
            with open(f"fold_{fold}/{feature}.pickle", "wb") as f:
                pickle.dump(transformer.named_transformers_[feature], f)

        fold_assignments[idx_va] = fold

    # save fold assignments
    pd.Series(fold_assignments, index=X.index, name="fold").to_csv("fold_assignments.csv")


if __name__ == "__main__":
    main()
