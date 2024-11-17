import pickle
from pathlib import Path

from omegaconf import DictConfig, ListConfig


def to_primitive(obj):  # type: ignore
    if isinstance(obj, DictConfig):
        return {k: to_primitive(v) for k, v in obj.items()}
    elif isinstance(obj, ListConfig):
        return [to_primitive(x) for x in obj]
    return obj


def save_pickle(obj: object, filepath: str | Path) -> None:
    with open(filepath, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(filepath: str | Path) -> object:
    with open(filepath, "rb") as file:
        return pickle.load(file)
