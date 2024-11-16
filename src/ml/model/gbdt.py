import pickle
from abc import abstractmethod
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoost, Pool
from typing_extensions import Self

from .base import BaseModel


class GBDTBaseModel(BaseModel):
    def save(self, filepath: str | Path) -> None:
        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        with open(filepath, "rb") as file:
            model = pickle.load(file)

        if not isinstance(model, cls):
            raise TypeError(
                f"Loaded object type does not match expected type. "
                f"Expected: {cls.__name__}, Actual: {type(model).__name__}"
            )

        return model

    @property
    @abstractmethod
    def feature_importance(self) -> pd.DataFrame:
        pass


class LightGBMModel(GBDTBaseModel):
    def __init__(
        self,
        objective: str,
        metric: str,
        boosting_type: str,
        num_leaves: int,
        learning_rate: float,
        feature_fraction: float,
        bagging_fraction: float,
        bagging_freq: int,
        max_bin: int,
        num_boost_round: int,
        early_stopping_rounds: int,
        device: str,
    ) -> None:
        self._model: lgb.Booster | None = None
        self._params = {
            "objective": objective,
            "metric": metric,
            "boosting_type": boosting_type,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "max_bin": max_bin,
            "device": device,
        }
        self._num_boost_round = num_boost_round
        self._early_stopping_rounds = early_stopping_rounds

    def fit(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray
    ) -> Self:
        data_tr, data_va = lgb.Dataset(X_tr, y_tr), lgb.Dataset(X_va, y_va)
        valid_sets = [data_tr, data_va]
        valid_names = ["train", "valid"]

        self._model = lgb.train(
            params=self._params,
            train_set=data_tr,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=self._num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self._early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model has not been trained.")
        return self._model.predict(X, num_iteration=self._model.best_iteration)  # type: ignore

    @property
    def feature_importance(self) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("Model has not been trained.")
        return pd.DataFrame(
            {
                "feature": self._model.feature_name(),
                "importance": self._model.feature_importance(),
            }
        )


class XGBoostModel(GBDTBaseModel):
    def __init__(
        self,
        objective: str,
        eval_metric: str,
        booster: str,
        learning_rate: float,
        min_split_loss: float,
        max_depth: int,
        min_child_weight: float,
        subsample: float,
        reg_lambda: float,
        reg_alpha: float,
        colsample_bytree: float,
        colsample_bylevel: float,
        num_boost_round: int,
        early_stopping_rounds: int,
        device: str,
    ) -> None:
        self._model: xgb.Booster | None = None
        self._params = {
            "objective": objective,
            "eval_metric": eval_metric,
            "booster": booster,
            "learning_rate": learning_rate,
            "min_split_loss": min_split_loss,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "colsample_bytree": colsample_bytree,
            "colsample_bylevel": colsample_bylevel,
            "device": device,
        }
        self._num_boost_round = num_boost_round
        self._early_stopping_rounds = early_stopping_rounds

    def fit(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray
    ) -> Self:
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        evals = [(dtrain, "train"), (dvalid, "valid")]

        self._model = xgb.train(
            params=self._params,
            dtrain=dtrain,
            evals=evals,
            num_boost_round=self._num_boost_round,
            early_stopping_rounds=self._early_stopping_rounds,
            verbose_eval=100,
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model has not been trained.")
        dtest = xgb.DMatrix(X, enable_categorical=True)
        return self._model.predict(dtest)

    @property
    def feature_importance(self) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("Model has not been trained.")
        importances = self._model.get_score(importance_type="total_gain")
        return pd.DataFrame(
            {
                "feature": list(importances.keys()),
                "importance": list(importances.values()),
            }
        )


class CatBoostModel(GBDTBaseModel):
    def __init__(
        self,
        loss_function: str,
        eval_metric: str,
        learning_rate: float,
        depth: int,
        l2_leaf_reg: float,
        random_strength: float,
        bagging_temperature: float,
        od_type: str,
        od_wait: int,
        iterations: int,
        early_stopping_rounds: int,
        task_type: str,
    ) -> None:
        self._model: CatBoost | None = None
        self._params = {
            "loss_function": loss_function,
            "eval_metric": eval_metric,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "random_strength": random_strength,
            "bagging_temperature": bagging_temperature,
            "od_type": od_type,
            "od_wait": od_wait,
            "iterations": iterations,
            "task_type": task_type,
        }
        self._early_stopping_rounds = early_stopping_rounds

    def fit(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray
    ) -> Self:
        cat_features = X_tr.select_dtypes(include=["category"]).columns.tolist()
        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        eval_pool = Pool(X_va, y_va, cat_features=cat_features)

        self._model = CatBoost(self._params)
        self._model.fit(
            train_pool,
            eval_set=eval_pool,
            use_best_model=True,
            early_stopping_rounds=self._early_stopping_rounds,
            verbose=100,
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model has not been trained.")

        cat_features = X.select_dtypes(include=["category"]).columns.tolist()
        test_pool = Pool(X, cat_features=cat_features)
        return self._model.predict(test_pool, prediction_type="RawFormulaVal")

    @property
    def feature_importance(self) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("Model has not been trained.")
        return pd.DataFrame(
            {
                "feature": self._model.feature_names_,
                "importance": self._model.feature_importances_,
            }
        )
