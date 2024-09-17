from pathlib import Path

from .base import BaseConfig, BaseModel
from .gbdt import LightGBMModel


class ModelFactory:
    def __init__(self, model_type: str, config: BaseConfig) -> None:
        self._config = config

        if model_type == "lightgbm":
            self._model_cls = LightGBMModel
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def build(self) -> BaseModel:
        return self._model_cls(self._config)  # type: ignore

    @staticmethod
    def load(file_path: str | Path) -> BaseModel:
        return BaseModel.load(file_path)
