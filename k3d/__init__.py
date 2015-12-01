from ipywidgets import DOMWidget
from IPython.display import display
from traitlets import Unicode, Bytes, Dict
from .factory import Factory
from .objects import Drawable
from .queue import Queue
from .version import version  # noqa, pylint: disable=no-name-in-module
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
        self.__update_strategy = self.__pass

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

    def update(self, obj_id, attr):
        self.__update_strategy(obj_id, attr)

    def __update(self, obj_id, attr):
        data = {
            'id': obj_id,
            'attr': attr,
        }

        self.data = self.__to_compressed_base64(self.__to_json(data))

    def __show(self, objs):
        for obj in objs:
            self.data = self.__to_compressed_base64(self.__to_json(obj.data))

    @staticmethod
    def __to_json(data):
        return json.dumps(data, separators=(',', ':')).encode(encoding='ascii')

    @classmethod
    def __to_compressed_base64(cls, data):
        return base64.b64encode(zlib.compress(data, cls.COMPRESSION_LEVEL))

    def display(self):
        display(self)

    def _ipython_display_(self, **kwargs):
        self.__display_strategy(**kwargs)

    def __display(self, **kwargs):
        super(K3D, self)._ipython_display_(**kwargs)

        self.__queue.flush()
        self.__display_strategy = self.__pass
        self.__update_strategy = self.__update

    def __pass(self, *args, **kwargs):
        pass
