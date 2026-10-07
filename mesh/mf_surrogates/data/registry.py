from typing import Optional, Callable
from torch.utils.data import Dataset


DATASET_REGISTRY: dict[str, type[Dataset]] = {}

def register_dataset(name: Optional[str] = None) -> Callable[[type[Dataset]], type[Dataset]]:
    def decorator(cls: type[Dataset]) -> type[Dataset]:
        reg_name = name or cls.__name__
        DATASET_REGISTRY[reg_name] = cls
        return cls
    return decorator

def get_dataset_class(name: str) -> type[Dataset]:
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' is not registered. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]
