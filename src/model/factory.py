from pathlib import Path
from typing import Any, Literal

from .base import BaseModel
from .gbdt import CatBoostModel, LightGBMModel, XGBoostModel
from .nn import NNModel


class ModelFactory:
    @staticmethod
    def build(
        model_type: Literal["lightgbm", "xgboost", "catboost", "nn"],
        **params: Any,
    ) -> BaseModel:
        if model_type == "lightgbm":
            return LightGBMModel(**params)
        elif model_type == "xgboost":
            return XGBoostModel(**params)
        elif model_type == "catboost":
            return CatBoostModel(**params)
        elif model_type == "nn":
            return NNModel(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def load(file_path: str | Path) -> BaseModel:
        return BaseModel.load(file_path)
