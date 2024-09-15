from logging import getLogger

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from sklearn.metrics import mean_squared_error

from config import Config
from ml.cv import run_group_cv
from ml.model.factory import ModelFactory
from process.process import preprocess

logger = getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(config: Config) -> None:
    # preprocess
    dataset = pl.read_csv(config.data_path).head(1000)
    X, y, groups = preprocess(dataset)

    # train & evaluate
    model_factory = ModelFactory(config.model_type, dict(config.model_params))  # type: ignore
    models, oof = run_group_cv(X, y, model_factory, groups)  # type: ignore

    rmse = mean_squared_error(y, oof, squared=False)
    logger.info(f"mse: {rmse}")

    for fold, model in enumerate(models, start=1):
        model.save(f"model_{fold}.pickle")


if __name__ == "__main__":
    main()
