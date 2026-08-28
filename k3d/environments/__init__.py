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


def save_js(path):
    """Write a sideload script defining window.k3dEnvironments.

    Included on a page (once, next to standalone.js), it lets kernel-less
    contexts - HTML snapshots, documentation embeds - realise the photographic
    catalog names, which otherwise degrade to the neutral preset. The maps
    travel as zlib-compressed float16, ~1 MB for the whole catalog.
    """
    import base64
    import json
    import zlib

    entries = {}
    for name in available():
        with np.load(os.path.join(_DIR, name + ".npz")) as f:
            half = f["map"].astype(np.float16)
        entries[name] = {
            "b64": base64.b64encode(zlib.compress(half.tobytes(), 9)).decode("ascii"),
            "shape": list(half.shape),
        }

    with open(path, "w", encoding="utf-8") as out:
        out.write("window.k3dEnvironments = ")
        json.dump(entries, out)
        out.write(";\n")
