from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from typing_extensions import Self

from .base import BaseConfig, BaseModel


@dataclass
class LightGBMConfig(BaseConfig):
    objective: str
    metric: str
    boosting_type: str
    num_leaves: int
    learning_rate: float
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    num_boost_round: int
    early_stopping_rounds: int
    device: str | None = None


class LightGBMModel(BaseModel):
    def __init__(self, config: LightGBMConfig) -> None:
        super().__init__(config)
        self._model: lgb.Booster | None = None

    def fit(
        self, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray
    ) -> Self:
        params = self._params.copy()
        num_boost_round = params.pop("num_boost_round")
        early_stopping_rounds = params.pop("early_stopping_rounds")

        data_tr, data_va = lgb.Dataset(X_tr, y_tr), lgb.Dataset(X_va, y_va)
        valid_sets = [data_tr, data_va]
        valid_names = ["train", "valid"]

        self._model = lgb.train(
            params=params,
            train_set=data_tr,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model has not been trained.")
        return self._model.predict(X, num_iteration=self._model.best_iteration)  # type: ignore
