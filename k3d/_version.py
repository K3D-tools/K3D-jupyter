import json
import importlib.resources

__all__ = ["__version__"]


package_json_path = (
    importlib.resources.files(__package__)
    / "labextension"
    / "package.json"
)
__version__ = json.loads(package_json_path.read_text())['version']
