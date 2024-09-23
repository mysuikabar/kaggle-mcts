import glob
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .model.base import BaseModel
from .model.factory import ModelFactory

logger = getLogger(__name__)


def run_group_cv(
    model_factory: ModelFactory,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> tuple[list[BaseModel], np.ndarray, np.ndarray]:
    """
    Performs group-based cross-validation and returns models, out-of-fold predictions, and fold assignments.
    """
    assert len(X) == len(y) == len(groups)

    models: list[BaseModel] = []
    oof = np.zeros(len(X))
    fold_assignments = np.zeros(len(X), dtype=int)

    kf = GroupKFold(n_splits)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(X, y, groups), start=1):
        logger.info(f"Training fold {fold}")

        X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
        X_va, y_va = X.iloc[idx_va], y[idx_va]

        model = model_factory.build()
        model.fit(X_tr, y_tr, X_va, y_va)

        oof[idx_va] = model.predict(X_va)
        fold_assignments[idx_va] = fold
        models.append(model)

    return models, oof, fold_assignments


def load_models(model_dir: Path) -> list[BaseModel]:
    return [ModelFactory.load(path) for path in glob.glob(str(model_dir / "*.pickle"))]


def predict_models(models: list[BaseModel], X: np.ndarray) -> np.ndarray:
    return np.stack([model.predict(X) for model in models]).mean(axis=0)
