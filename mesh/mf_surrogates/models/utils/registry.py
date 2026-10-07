from typing import Optional


MODEL_REGISTRY = {}

def register_model(name: Optional[str] = None):
    def decorator(fn):
        nonlocal name
        if name is None:
            name = fn.__name__
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator

def get_model_class(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' is not registered. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]
