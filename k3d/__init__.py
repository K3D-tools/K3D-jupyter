from ipywidgets import DOMWidget
from IPython.display import display
from traitlets import Unicode, Bytes, Dict
from .objects import Objects
from .factory import Factory
import base64, json, zlib


class K3D(DOMWidget, Factory):
    _view_module = Unicode('nbextensions/k3d_widget/view', sync=True)
    _view_name = Unicode('K3DView', sync=True)
    _model_module = Unicode('nbextensions/k3d_widget/model', sync=True)
    _model_name = Unicode('K3DModel', sync=True)

    COMPRESSION_LEVEL = 1

    data = Bytes(sync=True)
    parameters = Dict(sync=True)

    def __init__(self, antialias=False, background_color=0xFFFFFF, height=512):
        super(K3D, self).__init__()

        self.__objects = Objects(self.__show)
        self.on_displayed(lambda x: self.__objects.flush())

        self.parameters = {
            'antialias': antialias,
            'backgroundColor': background_color,
            'height': height,
        }

    def __add__(self, obj):
        self.__objects.add(obj)
        return self

    def display(self):
        display(self)

    def __show(self, obj):
        self.data = base64.b64encode(zlib.compress(json.dumps(obj, separators=(',', ':')), self.COMPRESSION_LEVEL))
