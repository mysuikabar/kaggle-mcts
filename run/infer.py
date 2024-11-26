import os
import sys
from dataclasses import dataclass
from logging import StreamHandler, getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

import model  # noqa: F401
import process  # noqa: F401
from consts import REPO_ROOT
from model.gbdt import GBDTBaseModel
from model.nn import NNModel
from process.feature import FeatureProcessor
from process.pipeline import PreprocessPipeline, postprocess
from process.text import ari, clri, mcalpine_eflaw  # noqa: F401
from process.transformers import TabularDataTransformer
from utils.helper import load_pickle

# overwrite here
MODEL_CONFIGS = [
    {"model_type": "catboost", "dataset": "run_name", "weight": 1},
]


@dataclass
class Config:
    input_dir: Path
    test_path: Path
    submission_path: Path
    evaluation_api_path: Path | None = None


config_local_env = Config(
    input_dir=REPO_ROOT / "outputs",
    test_path=REPO_ROOT / "data/raw/test.csv",
    submission_path=REPO_ROOT / "data/raw/sample_submission.csv",
    evaluation_api_path=REPO_ROOT / "data/raw",
)

config_kaggle_env = Config(
    input_dir=Path("/kaggle/input/"),
    test_path=Path("/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv"),
    submission_path=Path("/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv"),
)

if os.getenv("LOCAL_ENVIRONMENT"):
    config = config_local_env
    sys.path.append(str(config.evaluation_api_path))
else:
    config = config_kaggle_env

import kaggle_evaluation.mcts_inference_server  # noqa: E402

logger = getLogger(__name__)
logger.addHandler(StreamHandler(sys.stdout))


def predict(test: pl.DataFrame, submission: pl.DataFrame) -> pl.DataFrame:
    predictions: list[np.ndarray] = []

    for model_config in MODEL_CONFIGS:
        logger.info(f"Predicting with {model_config}")

        model_type = model_config["model_type"]
        dataset_dir = config.input_dir / model_config["dataset"]  # type: ignore
        weight = model_config["weight"]

        # feature engineering
        feature_processor: FeatureProcessor = load_pickle(dataset_dir / "feature_processor.pickle")
        feature_processor.disable_feature_store()
        test_processed = feature_processor.transform(test.to_pandas())

        preds = []

        for dir_path in tqdm(dataset_dir.glob("fold_*")):
            # process test data
            pipeline: PreprocessPipeline = load_pickle(dir_path / "pipeline.pickle")
            features = pd.read_csv(dir_path / "features.csv")["0"].tolist()
            X = pipeline.transform(test_processed).filter(features)

            # load model
            if model_type != "nn":
                model = GBDTBaseModel.load(dir_path / "model.pickle")
            else:
                transformer: TabularDataTransformer = load_pickle(dir_path / "transformer.pickle")
                X = transformer.transform(X)
                model = NNModel.load(dir_path / "model.pth")

            # predict
            pred = postprocess(model.predict(X))
            preds.append(pred)

        prediction = np.stack(preds).mean(axis=0)
        predictions.append(prediction * weight)

    ensemble_prediction = np.stack(predictions).sum(axis=0)
    return submission.with_columns(pl.Series("utility_agent1", ensemble_prediction))


def main() -> None:
    inference_server = kaggle_evaluation.mcts_inference_server.MCTSInferenceServer(
        predict  # type: ignore
    )

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    else:
        inference_server.run_local_gateway((config.test_path, config.submission_path))


if __name__ == "__main__":
    main()
