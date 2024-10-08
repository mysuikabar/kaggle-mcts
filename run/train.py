from logging import getLogger
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import wandb
from config.train import Config
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from features import feature_expressions_master
from metric import calculate_metrics
from ml.model.factory import ModelFactory
from process.feature import FeatureProcessor
from process.process import PreProcessor
from process.text import TfidfProcessor
from utils.seed import seed_everything

logger = getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def load_processor(tfidf_dir: Path) -> PreProcessor:
    col2tfidf: dict[str, TfidfProcessor] = {}
    for path in tfidf_dir.iterdir():
        col2tfidf[path.stem] = TfidfProcessor.load(path)
    return PreProcessor(col2tfidf)


@hydra.main(version_base=None, config_name="config")
def main(config: Config) -> None:
    seed_everything(config.seed)

    # load data
    logger.info("Loading data")
    df = pd.read_csv(config.data_path)
    X = df.drop(columns=[config.target])
    y = df[config.target].values
    logger.info(f"Raw data shape: {df.shape}")

    # feature engineering
    logger.info("Feature engineering")
    features = feature_expressions_master.filter(config.feature.use_features)
    feature_processor = FeatureProcessor(features, config.feature.feature_store_dir)
    X = feature_processor.transform(X)
    feature_processor.save("feature_processor.pickle")
    logger.info(f"Feature added data shape: {X.shape}")

    # cross validation
    model_factory = ModelFactory(config.model.type, config.model.config)
    oof = np.zeros(len(y))
    fold_assignments = pd.read_csv(
        config.preprocess_dir / "fold_assignments.csv", index_col=0
    )["fold"]

    for fold in sorted(fold_assignments.unique()):
        logger.info(f"Training fold {fold}")

        idx_tr = fold_assignments[fold_assignments != fold].index
        idx_va = fold_assignments[fold_assignments == fold].index
        X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
        X_va, y_va = X.iloc[idx_va], y[idx_va]

        # preprocess
        processor = load_processor(config.preprocess_dir / f"fold_{fold}")
        X_tr = processor.fit_transform(X_tr)
        X_va = processor.transform(X_va)
        logger.info(f"Processed data shape: {X_tr.shape}")

        # train & predict
        model = model_factory.build()
        model.fit(X_tr, y_tr, X_va, y_va)
        oof[idx_va] = model.predict(X_va)

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
