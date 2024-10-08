from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoost, Pool
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
    max_bin: int
    device: str | None = None


class LightGBMModel(BaseModel):
    def __init__(self, config: LightGBMConfig) -> None:
        super().__init__(config)
        self._model: lgb.Booster | None = None

    def fit(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray
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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model has not been trained.")
        return self._model.predict(X, num_iteration=self._model.best_iteration)  # type: ignore


@dataclass
class XGBoostConfig(BaseConfig):
    objective: str
    eval_metric: str
    booster: str
    learning_rate: float
    min_split_loss: float
    max_depth: int
    min_child_weight: float
    subsample: float
    reg_lambda: float
    reg_alpha: float
    colsample_bytree: float
    colsample_bylevel: float
    num_boost_round: int
    early_stopping_rounds: int
    device: str | None = None


class XGBoostModel(BaseModel):
    def __init__(self, config: XGBoostConfig) -> None:
        super().__init__(config)
        self._model: xgb.Booster | None = None

    def fit(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray
    ) -> Self:
        params = self._params.copy()
        num_boost_round = params.pop("num_boost_round")
        early_stopping_rounds = params.pop("early_stopping_rounds")

        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        evals = [(dtrain, "train"), (dvalid, "valid")]

        self._model = xgb.train(
            params=params,
            dtrain=dtrain,
            evals=evals,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100,
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model has not been trained.")
        dtest = xgb.DMatrix(X, enable_categorical=True)
        return self._model.predict(dtest)


@dataclass
class CatBoostConfig(BaseConfig):
    loss_function: str
    eval_metric: str
    learning_rate: float
    depth: int
    l2_leaf_reg: float
    random_strength: float
    bagging_temperature: float
    od_type: str
    od_wait: int
    iterations: int
    early_stopping_rounds: int
    task_type: str | None = None


class CatBoostModel(BaseModel):
    def __init__(self, config: CatBoostConfig) -> None:
        super().__init__(config)
        self._model: CatBoost | None = None

    def fit(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray
    ) -> Self:
        params = self._params.copy()
        early_stopping_rounds = params.pop("early_stopping_rounds")
        cat_features = X_tr.select_dtypes(include=["category"]).columns.tolist()

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        eval_pool = Pool(X_va, y_va, cat_features=cat_features)

        self._model = CatBoost(params)
        self._model.fit(
            train_pool,
            eval_set=eval_pool,
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
            verbose=100,
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model has not been trained.")

        cat_features = X.select_dtypes(include=["category"]).columns.tolist()
        test_pool = Pool(X, cat_features=cat_features)
        return self._model.predict(test_pool, prediction_type="RawFormulaVal")
