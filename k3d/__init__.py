from ipywidgets import DOMWidget
from IPython.display import display
from traitlets import Unicode, Bytes, Dict, Bool
from functools import partial
from .factory import Factory
from .objects import Drawable
from .queue import Queue
from .version import version
import base64
import json
import zlib


class K3D(DOMWidget, Factory):
    _view_module = Unicode('nbextensions/k3d_widget/view', sync=True)
    _view_name = Unicode('K3DView', sync=True)
    _model_module = Unicode('nbextensions/k3d_widget/model', sync=True)
    _model_name = Unicode('K3DModel', sync=True)
    version = version

    COMPRESSION_LEVEL = 1

    camera_auto_fit = Bool(sync=True)
    data = Bytes(sync=True)
    parameters = Dict(sync=True)

    def __init__(self, antialias=False, background_color=0xFFFFFF, camera_auto_fit=True, height=512):
        super(K3D, self).__init__()
        self.on_msg(self.__on_msg)

        self.__queue = Queue(self.__show)
        self.__display_strategy = self.__display
        self.__update_strategy = self.__pass
        self.__on_msg_strategy = self.__pass

        self.camera_auto_fit = camera_auto_fit
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

    def __sub__(self, objs):
        assert isinstance(objs, Drawable)

        for obj in objs:
            if obj.unset_plot(self):
                self.__update_strategy(obj.id)

        return self

    def fetch_data(self, obj):
        self.__on_msg_strategy = partial(self.__on_data, obj=obj)
        self.send(obj.id)

    def __on_msg(self, *args):
        self.__on_msg_strategy(args[1])

    def __on_data(self, data, obj):
        self.__on_msg_strategy = self.__pass
        obj.update(data)

    def update(self, obj_id, attr):
        self.__update_strategy(obj_id, attr)

    def __update(self, obj_id, attr=None):
        data = {
            'id': obj_id,
            'attr': attr,
        }

        if attr is None:
            del data['attr']

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
