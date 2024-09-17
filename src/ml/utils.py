from dataclasses import asdict, is_dataclass
from typing import Any


def to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    elif is_dataclass(obj):
        return asdict(obj)  # type: ignore
    else:
        try:
            return dict(obj)
        except Exception:
            raise ValueError(
                "Unable to convert object to dictionary. Please use a dictionary, dataclass, or an object that can be converted to a dictionary."
            )
