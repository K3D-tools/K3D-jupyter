import json
from pathlib import Path

__all__ = ["__version__"]


def _fetchVersion() -> str:
    # installed package: the distribution metadata is the source of truth
    try:
        from importlib.metadata import version

        return version("k3d")
    except Exception:
        pass

    # source tree: js/package.json carries the canonical version
    js_pkg = Path(__file__).parent.parent / "js" / "package.json"
    if js_pkg.exists():
        try:
            with js_pkg.open() as f:
                return json.load(f)["version"]
        except (FileNotFoundError, KeyError):
            pass

    return "0.0.0"


__version__ = _fetchVersion()
