import os
from dataclasses import dataclass
from pathlib import Path

from consts import REPO_ROOT


@dataclass
class Config:
    processor_path: Path
    model_dir: Path
    test_path: Path
    submission_path: Path
    evaluation_api_path: Path | None = None


_config_local_env = Config(
    processor_path=REPO_ROOT / "outputs" / "trial" / "processor.pickle",
    model_dir=REPO_ROOT / "outputs" / "trial" / "models",
    test_path=REPO_ROOT / "data" / "raw" / "test.csv",
    submission_path=REPO_ROOT / "data" / "raw" / "sample_submission.csv",
    evaluation_api_path=REPO_ROOT / "data" / "raw",
)

_config_kaggle_env = Config(
    processor_path=Path("/kaggle/input/mcts-models/processor.pickle"),
    model_dir=Path("/kaggle/input/mcts-models/models"),
    test_path=Path("/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv"),
    submission_path=Path(
        "/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv"
    ),
)


if os.getenv("LOCAL_ENVIRONMENT"):
    config = _config_local_env
else:
    config = _config_kaggle_env
