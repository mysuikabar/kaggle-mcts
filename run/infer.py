import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import model  # noqa: F401
import process  # noqa: F401
from consts import REPO_ROOT
from model.gbdt import GBDTBaseModel
from model.nn import NNModel
from process.feature import FeatureProcessor
from process.pipeline import PreprocessPipeline, postprocess
from process.transformers import TabularDataTransformer
from utils.helper import load_pickle

# overwrite these variables
MODEL_TYPE = "catboost"  # "catboost" | "lightgbm" | "xgboost" | "nn"
DATASET = "run_name"


@dataclass
class Config:
    dataset_dir: Path
    test_path: Path
    submission_path: Path
    model_type: str = MODEL_TYPE
    evaluation_api_path: Path | None = None


config_local_env = Config(
    dataset_dir=REPO_ROOT / f"outputs/{DATASET}",
    test_path=REPO_ROOT / "data/raw/test.csv",
    submission_path=REPO_ROOT / "data/raw/sample_submission.csv",
    evaluation_api_path=REPO_ROOT / "data/raw",
)

config_kaggle_env = Config(
    dataset_dir=Path(f"/kaggle/input/{DATASET}"),
    test_path=Path("/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv"),
    submission_path=Path("/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv"),
)

if os.getenv("LOCAL_ENVIRONMENT"):
    config = config_local_env
    sys.path.append(str(config.evaluation_api_path))
else:
    config = config_kaggle_env

import kaggle_evaluation.mcts_inference_server  # noqa: E402


def predict(test: pl.DataFrame, submission: pl.DataFrame) -> pl.DataFrame:
    # feature engineering
    feature_processor: FeatureProcessor = load_pickle(config.dataset_dir / "feature_processor.pickle")
    feature_processor.disable_feature_store()
    test = feature_processor.transform(test.to_pandas())

    preds = []

    for dir_path in config.dataset_dir.glob("fold_*"):
        # process test data
        pipeline: PreprocessPipeline = load_pickle(dir_path / "pipeline.pickle")
        features = pd.read_csv(dir_path / "features.csv")["0"].tolist()
        X = pipeline.transform(test).filter(features)

        # load model
        if config.model_type != "nn":
            model = GBDTBaseModel.load(dir_path / "model.pickle")
        else:
            transformer: TabularDataTransformer = load_pickle(dir_path / "transformer.pickle")
            X = transformer.transform(X)
            model = NNModel.load(dir_path / "model.pth")

        # predict
        pred = model.predict(X)
        pred = postprocess(pred)
        preds.append(pred)

    prediction = np.stack(preds).mean(axis=0)

    return submission.with_columns(pl.Series("utility_agent1", prediction))


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
