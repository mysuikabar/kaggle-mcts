import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from consts import REPO_ROOT
from ml.model.base import BaseModel
from ml.model.factory import ModelFactory
from process.process import Preprocessor

sys.path.append(str(REPO_ROOT / "data" / "raw"))
import kaggle_evaluation.mcts_inference_server  # type: ignore


@dataclass
class Config:
    processor_path: Path = REPO_ROOT / "outputs" / "trial" / "processor.pickle"
    model_dir: Path = REPO_ROOT / "outputs" / "trial" / "models"
    test_path: Path = REPO_ROOT / "data" / "raw" / "test.csv"
    submission_path: Path = REPO_ROOT / "data" / "raw" / "sample_submission.csv"


config = Config()


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
