from ipywidgets import DOMWidget
from IPython.display import display
from traitlets import Unicode, Bytes, Dict
from .factory import Factory
from .objects import Drawable
from .queue import Queue
import base64
import json
import zlib


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

        self.__queue = Queue(self.__show)
        self.__display_strategy = self.__display
        self.on_displayed(lambda x: self.__queue.flush())

        self.parameters = {
            'antialias': antialias,
            'backgroundColor': background_color,
            'height': height,
        }

    def __add__(self, obj):
        assert isinstance(obj, Drawable)

        if obj.set_plot(self):
            self.__queue.add(obj)

        return self

    def display(self):
        self.__display_strategy()

    def __display(self):
        display(self)
        self.__display_strategy = self.__pass

    def __show(self, objs):
        for obj in objs:
            self.data = base64.b64encode(zlib.compress(json.dumps(obj.__dict__, separators=(',', ':')), self.COMPRESSION_LEVEL))

    def __pass(self):
        pass
