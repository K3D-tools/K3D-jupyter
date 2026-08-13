"""Built-in photographic environments (Poly Haven, CC0), stored as float16 equirects."""

import os
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))


def available():
    """Names accepted by plot.environment on top of the procedural presets."""
    return sorted(f[:-4] for f in os.listdir(_DIR) if f.endswith(".npz"))


def load(name):
    """The (H, W, 3) float32 equirect for a catalog name, or None if unknown."""
    path = os.path.join(_DIR, name + ".npz")
    if not os.path.isfile(path):
        return None
    with np.load(path) as f:
        return f["map"].astype(np.float32)
