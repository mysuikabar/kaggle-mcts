from logging import getLogger

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from sklearn.metrics import mean_squared_error

from config import Config
from ml.cv import run_group_cv
from ml.model.factory import ModelFactory
from process.process import Preprocessor

logger = getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(config: Config) -> None:
    # preprocess
    df = pl.read_csv(config.data_path).head(1000)
    y = df.select("utility_agent1").to_numpy().ravel()

    processor = Preprocessor()
    X, groups = processor.fit_transform(df), processor.group_label
    processor.save("processor.pickle")

    # train & evaluate
    model_factory = ModelFactory(config.model_type, config.model_config)
    models, oof = run_group_cv(X, y, model_factory, groups)

    rmse = mean_squared_error(y, oof, squared=False)
    logger.info(f"mse: {rmse}")

    config.model_output_dir.mkdir(exist_ok=True)
    for fold, model in enumerate(models, start=1):
        model.save(config.model_output_dir / f"model_{fold}.pickle")


if __name__ == "__main__":
    main()
