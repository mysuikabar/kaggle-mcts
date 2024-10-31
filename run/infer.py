import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from config.infer import Config

from consts import REPO_ROOT
from ml.model.factory import ModelFactory
from process.feature import FeatureProcessor
from process.process import PreProcessor, postprocess

DATASET = "run_name"  # change here

config_local_env = Config(
    dataset_dir=REPO_ROOT / f"outputs/{DATASET}",
    test_path=REPO_ROOT / "data/raw/test.csv",
    submission_path=REPO_ROOT / "data/raw/sample_submission.csv",
    evaluation_api_path=REPO_ROOT / "data/raw",
)

config_kaggle_env = Config(
    dataset_dir=Path(f"/kaggle/input/{DATASET}"),
    test_path=Path("/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv"),
    submission_path=Path(
        "/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv"
    ),
)

if os.getenv("LOCAL_ENVIRONMENT"):
    config = config_local_env
    sys.path.append(str(config.evaluation_api_path))
else:
    config = config_kaggle_env

import kaggle_evaluation.mcts_inference_server  # noqa: E402


def predict(test: pl.DataFrame, submission: pl.DataFrame) -> pl.DataFrame:
    # feature engineering
    feature_processor = FeatureProcessor.load(
        config.dataset_dir / "feature_processor.pickle"
    )
    feature_processor.disable_feature_store()
    test = feature_processor.transform(test.to_pandas())

    preds = []

    for dir_path in config.dataset_dir.glob("fold_*"):
        # process test data
        processor = PreProcessor.load(dir_path / "processor.pickle")
        features = pd.read_csv(dir_path / "features.csv")["0"].tolist()
        X = processor.transform(test).filter(features)

        # predict
        model = ModelFactory.load(dir_path / "model.pickle")
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
