import ipywidgets as widgets
import numpy as np
from IPython.display import display
from traitlets import Unicode, Int
from traitlets import validate
from traittypes import Array
from typing import Any, Dict as TypingDict, List as TypingList, Optional, Union

from ._version import __version__ as version
from .colormaps import paraview_color_maps
from .helpers import array_serialization_wrap


class TF_editor(widgets.DOMWidget):
    _view_name = Unicode('TransferFunctionView').tag(sync=True)
    _model_name = Unicode('TransferFunctionModel').tag(sync=True)
    _view_module = Unicode('k3d').tag(sync=True)
    _model_module = Unicode('k3d').tag(sync=True)

    _view_module_version = Unicode(version).tag(sync=True)
    _model_module_version = Unicode(version).tag(sync=True)

    # readonly (specified at creation)
    height = Int().tag(sync=True)

    # read-write
    color_map = Array(dtype=np.float32).tag(sync=True, **array_serialization_wrap('color_map'))
    opacity_function = Array(dtype=np.float32).tag(sync=True,
                                                   **array_serialization_wrap('opacity_function'))

    def __init__(self, height: int, color_map: np.ndarray, opacity_function: np.ndarray, *args: Any,
                 **kwargs: Any) -> None:
        super(TF_editor, self).__init__()

        self.height = height

        with self.hold_trait_notifications():
            self.color_map = color_map
            self.opacity_function = opacity_function

        self.outputs: TypingList[widgets.Output] = []

    def display(self, **kwargs: Any) -> None:
        output = widgets.Output()

        with output:
            display(self, **kwargs)

        self.outputs.append(output)

        display(output)

    def close(self) -> None:
        for output in self.outputs:
            output.clear_output()

        self.outputs = []

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    @validate('color_map')
    def _validate_color_map(self, proposal: TypingDict[str, Any]) -> np.ndarray:
        if proposal['value'].shape == ():
            return proposal['value']

        cm_min, cm_max = np.min(proposal['value'][::4]), np.max(proposal['value'][::4])

        if cm_min != 0.0 or cm_max != 1.0:
            proposal['value'][::4] = (proposal['value'][::4] - cm_min) / (cm_max - cm_min)

        return proposal['value']

    @validate('opacity_function')
    def _validate_opacity_function(self, proposal: TypingDict[str, Any]) -> np.ndarray:
        if proposal['value'].shape == ():
            return proposal['value']

        of_min, of_max = np.min(proposal['value'][::2]), np.max(proposal['value'][::2])

        if of_min != 0.0 or of_max != 1.0:
            proposal['value'][::2] = (proposal['value'][::2] - of_min) / (of_max - of_min)

        return proposal['value']


def transfer_function_editor(color_map: np.ndarray = paraview_color_maps.Jet,
                             opacity_function: Optional[np.ndarray] = None,
                             height: int = 300) -> TF_editor:
    """Create a K3D Transfer function editor widget.

    Arguments:
        height: `int`.
            Height of the widget in pixels.
        color_map: `array`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        opacity_function: `array`.
            A list of float tuples (attribute value, opacity), sorted by attribute value. The first
            typles should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
    """
    if opacity_function is None:
        opacity_function = np.array([np.min(color_map[::4]), 0.0, np.max(color_map[::4]), 1.0],
                                    dtype=np.float32)

    return TF_editor(height, color_map, opacity_function)
