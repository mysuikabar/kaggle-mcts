from logging import getLogger
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars as pl
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from sklearn.model_selection import GroupKFold

from config.train import Config
from features import feature_expressions_master
from metric import calculate_metrics
from ml.model.factory import ModelFactory
from process.feature import FeatureProcessor
from process.process import Preprocessor
from utils.seed import seed_everything

logger = getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(config: Config) -> None:
    seed_everything(config.seed)

    # load data
    logger.info("Loading data")
    df = pd.read_csv(config.data_path)
    X = df.drop(columns=[config.target])
    y = df[config.target].values
    groups = df[config.groups].values
    logger.info(f"Raw data shape: {df.shape}")

    # feature engineering
    logger.info("Feature engineering")
    features = feature_expressions_master.filter(config.feature.use_features)
    feature_processor = FeatureProcessor(features, config.feature.feature_store_dir)
    X = feature_processor.run(pl.DataFrame(X)).to_pandas()
    logger.info(f"Feature engineered data shape: {X.shape}")

    # cross validation
    model_factory = ModelFactory(config.model.type, config.model.config)
    oof = np.zeros(len(y))
    fold_assignments = np.zeros(len(y), dtype=int)

    kf = GroupKFold(config.n_splits)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(X, y, groups), start=1):
        logger.info(f"Training fold {fold}")

        X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
        X_va, y_va = X.iloc[idx_va], y[idx_va]

        # preprocess
        processor = Preprocessor()
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
