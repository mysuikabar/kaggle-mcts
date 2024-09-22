import os
import sys

import polars as pl

from config.infer import config
from ml.cv import load_models, predict_models
from process.process import Preprocessor

if os.getenv("LOCAL_ENVIRONMENT"):
    sys.path.append(str(config.evaluation_api_path))

import kaggle_evaluation.mcts_inference_server  # type: ignore


def predict(test: pl.DataFrame, submission: pl.DataFrame) -> pl.DataFrame:
    processor = Preprocessor.load(config.processor_path)
    processor.disable_feature_store()
    X = processor.transform(test)

    models = load_models(config.model_dir)
    pred = predict_models(models, X)

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
