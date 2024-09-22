import glob
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

from config.infer import config
from ml.model.base import BaseModel
from ml.model.factory import ModelFactory
from process.process import Preprocessor

if os.getenv("LOCAL_ENVIRONMENT"):
    sys.path.append(str(config.evaluation_api_path))

import kaggle_evaluation.mcts_inference_server  # type: ignore


def load_models(model_dir: Path) -> list[BaseModel]:
    return [ModelFactory.load(path) for path in glob.glob(str(model_dir / "*.pickle"))]


def predict_models(X: np.ndarray, models: list[BaseModel]) -> np.ndarray:
    return np.stack([model.predict(X) for model in models]).mean(axis=0)


def predict(test: pl.DataFrame, submission: pl.DataFrame) -> pl.DataFrame:
    processor = Preprocessor.load(config.processor_path)
    processor.disable_feature_store()
    X = processor.transform(test)

    models = load_models(config.model_dir)
    pred = predict_models(X, models)

    return submission.with_columns(pl.Series("utility_agent1", pred))


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
