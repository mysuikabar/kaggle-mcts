from dataclasses import asdict, is_dataclass
from typing import Any

from omegaconf import DictConfig


def to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    elif is_dataclass(obj):
        return asdict(obj)  # type: ignore
    elif isinstance(obj, DictConfig):
        return dict(obj)
    else:
        raise TypeError(
            "Input must be a dictionary, an instance of a dataclass, or a DictConfig object."
        )
