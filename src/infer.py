import glob
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

from config.infer import Config
from consts import REPO_ROOT
from ml.model.factory import ModelFactory
from process.process import Preprocessor, postprocess

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
    import kaggle_evaluation.mcts_inference_server  # type: ignore
else:
    config = config_kaggle_env


def predict(test: pl.DataFrame, submission: pl.DataFrame) -> pl.DataFrame:
    preds = []
    for dir_path in glob.glob(str(config.dataset_dir / "fold_*")):
        # process test data
        processor = Preprocessor.load(f"{dir_path}/processor.pickle")
        processor.disable_feature_store()
        X = processor.transform(test)

        # predict
        model = ModelFactory.load(f"{dir_path}/model.pickle")
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
