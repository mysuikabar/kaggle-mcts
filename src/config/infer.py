from dataclasses import dataclass
from pathlib import Path

from consts import REPO_ROOT


@dataclass
class InferConfig:
    processor_path: Path = REPO_ROOT / "outputs" / "trial" / "processor.pickle"
    model_dir: Path = REPO_ROOT / "outputs" / "trial" / "models"
    test_path: Path = REPO_ROOT / "data" / "raw" / "test.csv"
    submission_path: Path = REPO_ROOT / "data" / "raw" / "sample_submission.csv"
