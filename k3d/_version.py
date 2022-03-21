import json
import os
from pathlib import Path

__all__ = ["__version__"]


def _fetchVersion():
    HERE = Path(os.path.abspath('../..'))

    for settings in HERE.rglob("package.json"):
        try:
            with settings.open() as f:
                return json.load(f)["version"]
        except FileNotFoundError:
            pass

    raise FileNotFoundError(f"Could not find package.json under dir {HERE!s}")


__version__ = _fetchVersion()
