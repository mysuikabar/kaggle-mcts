import pickle
from logging import getLogger
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import wandb
from conf.config import Config
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from features import feature_expressions_master
from metric import calculate_metrics, log_metrics
from model.factory import ModelFactory
from process.feature import FeatureProcessor, FeatureStore
from process.pipeline import PreprocessPipeline
from process.transformers import TabularDataTransformer
from utils.helper import to_primitive
from utils.seed import seed_everything

logger = getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def load_feature_processor(use_features: list[str], feature_store_dir: Path | None = None) -> FeatureProcessor:
    features = feature_expressions_master.filter(use_features)
    feature_store = FeatureStore(feature_store_dir) if feature_store_dir else None
    return FeatureProcessor(features, feature_store)


def load_process_pipeline(tfidf_dir: Path | None = None, use_columns: list[str] | None = None) -> PreprocessPipeline:
    if tfidf_dir is None:
        return PreprocessPipeline(use_columns=use_columns)

    col2tfidf = {}
    for path in tfidf_dir.iterdir():
        col2tfidf[path.stem] = pickle.load(open(path, "rb"))
    return PreprocessPipeline(col2tfidf=col2tfidf, use_columns=use_columns)


def filter_features_by_importance(importance: pd.DataFrame, num_features: int) -> list[str]:
    filtered_importance = importance.sort_values("importance", ascending=False).head(num_features)
    return filtered_importance["feature"].tolist()


@hydra.main(version_base=None, config_name="config")
def main(config: Config) -> None:
    seed_everything(config.seed)

    # load data
    df = pd.read_csv(config.data_path)
    X, y = df.drop(columns=[config.target]), df[config.target].values
    logger.info(f"Raw data shape: {df.shape}")

    # feature engineering
    feature_processor = load_feature_processor(config.feature.use_features, config.feature.store_dir)
    X = feature_processor.transform(X)
    feature_processor.save("feature_processor.pickle")
    logger.info(f"Feature added data shape: {X.shape}")

    # cross validation
    model_factory = ModelFactory()
    oof = np.zeros(len(y))
    fold_assignments = pd.read_csv(config.fold_assignment_path, index_col=0)["fold"]

    for fold in sorted(fold_assignments.unique()):
        logger.info(f"Training fold {fold}")
        output_dir = Path(f"fold_{fold}")
        output_dir.mkdir(exist_ok=True)

        # split
        idx_tr = fold_assignments[fold_assignments != fold].index
        idx_va = fold_assignments[fold_assignments == fold].index
        X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
        X_va, y_va = X.iloc[idx_va], y[idx_va]

        # preprocess
        features = None
        if config.importance_dir is not None:
            importance = pd.read_csv(config.importance_dir / f"fold_{fold}/importance.csv")
            features = filter_features_by_importance(importance, config.num_features)
        tfidf_dir = config.tfidf_dir / f"fold_{fold}" if config.tfidf_dir else None

        pipeline = load_process_pipeline(tfidf_dir=tfidf_dir, use_columns=features)
        X_tr, X_va = pipeline.fit_transform(X_tr), pipeline.transform(X_va)
        pipeline.save(output_dir / "pipeline.pickle")
        X_tr.columns.to_series().to_csv(output_dir / "features.csv", index=False)
        logger.info(f"Processed data shape: {X_tr.shape}")

        # train
        if config.model.type != "nn":
            model = model_factory.build(config.model.type, **config.model.config)
            model.fit(X_tr, y_tr, X_va, y_va)
            model.save(output_dir / "model.pickle")
            model.feature_importance.to_csv(output_dir / "importance.csv", index=False)
        else:
            transformer = TabularDataTransformer()
            X_tr, X_va = transformer.fit_transform(X_tr), transformer.transform(X_va)
            pickle.dump(transformer, open(output_dir / "model" / "transformer.pickle", "wb"))

            model = model_factory.build(
                config.model.type,
                num_numerical_features=len(transformer.numerical_columns_),
                categorical_feature_dims=transformer.n_categories_,
                **to_primitive(config.model.config),
            )
            model.fit(X_tr, y_tr, X_va, y_va)
            model.save(output_dir / "model")

        # predict
        oof[idx_va] = model.predict(X_va)

    # evaluate
    metrics = calculate_metrics(y, oof, fold_assignments)
    log_metrics(metrics, logger)

    # log to wandb
    if config.wandb.enable:
        wandb.init(
            project=config.wandb.project,
            name=config.wandb.name,
            notes=config.wandb.notes,
            tags=[config.model.type],
            config=OmegaConf.to_container(config),  # type: ignore
        )
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
