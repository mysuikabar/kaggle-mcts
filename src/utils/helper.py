import pickle
from pathlib import Path


def save_pickle(obj: object, filepath: str | Path) -> None:
    with open(filepath, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(filepath: str | Path) -> object:
    with open(filepath, "rb") as file:
        return pickle.load(file)
