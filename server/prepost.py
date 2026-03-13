import numpy as np


def ensure_nchw(x: np.ndarray) -> np.ndarray:
    """Convert input to NCHW if a single sample CHW is provided."""
    if x.ndim == 3:
        return x[np.newaxis, ...]
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected CHW or NCHW input, got shape={x.shape}")


def adapt_channels(x: np.ndarray, model_channels: int) -> np.ndarray:
    """Pad or truncate channels on NCHW input."""
    if x.ndim != 4:
        raise ValueError("adapt_channels expects NCHW input")
    n, c, h, w = x.shape
    if c == model_channels:
        return x
    if c < model_channels:
        pad = np.zeros((n, model_channels - c, h, w), dtype=x.dtype)
        return np.concatenate([x, pad], axis=1)
    return x[:, :model_channels, :, :]
