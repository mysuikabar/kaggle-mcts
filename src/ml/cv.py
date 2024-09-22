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
    X: pd.DataFrame,
    y: np.ndarray,
    model_factory: ModelFactory,
    groups: np.ndarray,
    n_splits: int = 5,
) -> tuple[list[BaseModel], np.ndarray]:
    assert len(X) == len(y) and len(y) == len(groups)

    models: list[BaseModel] = []
    oof = np.zeros(len(X))

    kf = GroupKFold(n_splits)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(X, y, groups), start=1):
        logger.info(f"Training fold {fold}")

        X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
        X_va, y_va = X.iloc[idx_va], y[idx_va]

        model = model_factory.build()
        model.fit(X_tr, y_tr, X_va, y_va)

        oof[idx_va] = model.predict(X_va)
        models.append(model)

    return models, oof


def load_models(model_dir: Path) -> list[BaseModel]:
    return [ModelFactory.load(path) for path in glob.glob(str(model_dir / "*.pickle"))]


def predict_models(models: list[BaseModel], X: np.ndarray) -> np.ndarray:
    return np.stack([model.predict(X) for model in models]).mean(axis=0)
