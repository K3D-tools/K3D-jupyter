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

    # source tree: the root package.json is what hatch-nodejs-version reads, so it is the one
    # the wheel and the PyPI release are named after
    js_pkg = Path(__file__).parent.parent / "package.json"
    if js_pkg.exists():
        try:
            with js_pkg.open() as f:
                return json.load(f)["version"]
        except (FileNotFoundError, KeyError):
            pass

    return "0.0.0"


__version__ = _fetchVersion()
