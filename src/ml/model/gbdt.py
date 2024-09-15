from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from ..utils import to_dict
from .base import BaseConfig, BaseModel


@dataclass
class LightGBMConfig(BaseConfig):
    objective: str = "regression"
    metric: str = "rmse"
    boosting_type: str = "gbdt"
    num_leaves: int = 30
    learning_rate: float = 0.05
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    num_boost_round: int = 1000
    early_stopping_rounds: int = 10


class LightGBMModel(BaseModel):
    def __init__(self, config: LightGBMConfig) -> None:
        super().__init__(config)
        self._model: lgb.Booster | None = None

    def fit(
        self, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray
    ) -> None:
        params = to_dict(self._config)
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model has not been trained.")
        return self._model.predict(X, num_iteration=self._model.best_iteration)  # type: ignore
