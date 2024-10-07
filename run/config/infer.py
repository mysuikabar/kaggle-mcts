from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    dataset_dir: Path
    test_path: Path
    submission_path: Path
    evaluation_api_path: Path | None = None
