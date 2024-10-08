from logging import getLogger
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from config.preprocess import Config
from hydra.core.config_store import ConfigStore
from sklearn.model_selection import GroupKFold

from features import feature_expressions_master
from process.feature import FeatureProcessor
from process.text import parallel_fit_tfidf
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
    groups = df[config.groups].values
    logger.info(f"Raw data shape: {df.shape}")

    # feature engineering
    logger.info("Feature engineering")
    feature_processor = FeatureProcessor(
        feature_expressions_master, config.feature_store_dir
    )
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
        tfidf = parallel_fit_tfidf(
            X_tr, config.tfidf.text_features, config.tfidf.max_features
        )

        # save
        Path(f"fold_{fold}").mkdir(exist_ok=True)
        for feature, processor in tfidf.items():
            processor.save(f"fold_{fold}/{feature}.pickle")

        fold_assignments[idx_va] = fold

    # save fold assignments
    pd.Series(fold_assignments, index=X.index, name="fold").to_csv(
        "fold_assignments.csv"
    )


if __name__ == "__main__":
    main()
