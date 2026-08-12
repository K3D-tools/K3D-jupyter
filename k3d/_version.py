import json
from pathlib import Path

__all__ = ["__version__"]


def _fetchVersion() -> str:
    HERE = Path(__file__).parent.resolve()

    # Search in k3d/labextension first
    labextension_pkg = HERE / "labextension" / "package.json"
    if labextension_pkg.exists():
        try:
            with labextension_pkg.open() as f:
                return json.load(f)["version"]
        except (FileNotFoundError, KeyError):
            pass

    # Fallback to searching locally
    for settings in HERE.rglob("package.json"):
        try:
            with settings.open() as f:
                return json.load(f)["version"]
        except (FileNotFoundError, KeyError):
            pass

    # Fallback: js/package.json (when running from source root)
    # k3d/../js/package.json
    js_pkg = HERE.parent / "js" / "package.json"
    if js_pkg.exists():
         try:
            with js_pkg.open() as f:
                return json.load(f)["version"]
         except (FileNotFoundError, KeyError):
            pass

    return "0.0.0"


__version__ = _fetchVersion()
