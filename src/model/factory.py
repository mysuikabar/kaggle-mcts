from .base import BaseModel
from .gbdt import LightGBMModel


class ModelFactory:
    def __init__(self, model_type: str, params: dict) -> None:
        self._params = params.copy()

        if model_type == "lightgbm":
            self._model_cls = LightGBMModel
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def build(self) -> BaseModel:
        params = self._params.copy()
        return self._model_cls(**params)
