from .base import BaseModel
from .gbdt import LightGBMModel


class ModelFactory:
    @staticmethod
    def create_model(model_type: str, params: dict) -> BaseModel:
        if model_type == "lightgbm":
            return LightGBMModel(params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
