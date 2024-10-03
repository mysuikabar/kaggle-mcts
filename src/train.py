from logging import getLogger
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from sklearn.model_selection import GroupKFold

from config.train import Config
from features import feature_expressions_master
from metric import calculate_metrics
from ml.model.factory import ModelFactory
from process.feature import FeatureProcessor, FeatureStore
from process.process import Preprocessor
from utils.seed import seed_everything

logger = getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def create_preprocessor(
    use_features: list[str], feature_store_dir: Path | None
) -> Preprocessor:
    feature_expressions = feature_expressions_master.filter(use_features)
    feature_store = FeatureStore(feature_store_dir) if feature_store_dir else None
    feature_processor = FeatureProcessor(feature_expressions, feature_store)

    return Preprocessor(feature_processor)


@hydra.main(version_base=None, config_name="config")
def main(config: Config) -> None:
    seed_everything(config.seed)

    df = pd.read_csv(config.data_path)
    X = df.drop(columns=[config.target])
    y = df[config.target].values
    groups = df[config.groups].values
    model_factory = ModelFactory(config.model.type, config.model.config)

    # cross validation
    oof = np.zeros(len(y))
    fold_assignments = np.zeros(len(y), dtype=int)

    kf = GroupKFold(config.n_splits)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(X, y, groups), start=1):
        logger.info(f"Training fold {fold}")

        X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
        X_va, y_va = X.iloc[idx_va], y[idx_va]

        # preprocess
        processor = create_preprocessor(**config.preprocess)  # type: ignore
        X_tr = processor.fit_transform(X_tr)
        X_va = processor.transform(X_va)

        # train
        model = model_factory.build()
        model.fit(X_tr, y_tr, X_va, y_va)

        # predict
        oof[idx_va] = model.predict(X_va)
        fold_assignments[idx_va] = fold

        # save
        Path(f"fold_{fold}").mkdir(exist_ok=True)
        processor.save(f"fold_{fold}/processor.pickle")
        model.save(f"fold_{fold}/model.pickle")

    # evaluate
    metrics = calculate_metrics(y, oof, fold_assignments)
    for metric, score in metrics.items():
        logger.info(f"{metric}: {score}")

    # log to wandb
    if config.wandb.enable:
        wandb.init(
            project=config.wandb.project,
            name=config.wandb.name,
            config=OmegaConf.to_container(config),  # type: ignore
        )
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
