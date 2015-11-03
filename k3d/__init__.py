from ipywidgets import DOMWidget
from IPython.display import display
from traitlets import Unicode, Bytes, Int
from .objects import Objects
from .factory import Factory
import base64
import zlib
import json


class K3D(DOMWidget, Factory):
    _view_module = Unicode('nbextensions/k3d_widget/view', sync=True)
    _view_name = Unicode('K3DView', sync=True)
    _model_module = Unicode('nbextensions/k3d_widget/model', sync=True)
    _model_name = Unicode('K3DModel', sync=True)

    COMPRESSION_LEVEL = 9

    height = Int(sync=True)
    data = Bytes(sync=True)

    def __init__(self, height=512):
        super(K3D, self).__init__()

        self.__objects = Objects(self.__show)
        self.height = height
        self.on_displayed(lambda x: self.__objects.flush())

    def __add__(self, obj):
        self.__objects.add(obj)
        return self

    def display(self):
        display(self)

    def __show(self, obj):
        self.data = base64.b64encode(zlib.compress(json.dumps(obj, separators=(',', ':')), self.COMPRESSION_LEVEL))
