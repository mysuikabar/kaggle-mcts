from omegaconf import DictConfig, ListConfig


def to_primitive(obj):  # type: ignore
    if isinstance(obj, DictConfig):
        return {k: to_primitive(v) for k, v in obj.items()}
    elif isinstance(obj, ListConfig):
        return [to_primitive(x) for x in obj]
    return obj
