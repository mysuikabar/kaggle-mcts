from logging import getLogger
from pathlib import Path

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from sklearn.metrics import mean_squared_error

from config.train import Config
from features import feature_expressions_master
from ml.cv import run_group_cv
from ml.model.factory import ModelFactory
from process.feature import FeatureProcessor, FeatureStore
from process.process import Preprocessor
from utils.seed import seed_everything

logger = getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def create_preprocessor(
    use_features: list[str], feature_store_dir: Path
) -> Preprocessor:
    feature_expressions = feature_expressions_master.filter(use_features)
    feature_store = FeatureStore(feature_store_dir)
    feature_processor = FeatureProcessor(feature_expressions, feature_store)

    return Preprocessor(feature_processor)


@hydra.main(version_base=None, config_name="config")
def main(config: Config) -> None:
    seed_everything(config.seed)

    # preprocess
    df = pl.read_csv(config.data_path).head(1000)
    y = df.select("utility_agent1").to_numpy().ravel()

    processor = create_preprocessor(**config.preprocess)  # type: ignore
    X, groups = processor.fit_transform(df), processor.group_label
    processor.save("processor.pickle")

    # train & evaluate
    model_factory = ModelFactory(config.model.type, config.model.config)
    models, oof = run_group_cv(X, y, model_factory, groups)

    rmse = mean_squared_error(y, oof, squared=False)
    logger.info(f"mse: {rmse}")

    config.model.output_dir.mkdir(exist_ok=True)
    for fold, model in enumerate(models, start=1):
        model.save(config.model.output_dir / f"model_{fold}.pickle")


if __name__ == "__main__":
    main()
