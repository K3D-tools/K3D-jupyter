import ipywidgets as widgets
from traitlets import Unicode, Bytes, Dict, Bool, Int

from functools import partial
from .factory import Factory
from .objects import Drawable
from ._version import version_info
import base64
import json
import zlib


class K3D(widgets.DOMWidget, Factory):
    version = version_info
    _view_name = Unicode('K3DView').tag(sync=True)
    _model_name = Unicode('K3DModel').tag(sync=True)
    _view_module = Unicode('k3d').tag(sync=True)
    _model_module = Unicode('k3d').tag(sync=True)

    _view_module_version = Unicode('^2.0.0').tag(sync=True)
    _model_module_version = Unicode('^2.0.0').tag(sync=True)

    camera_auto_fit = Bool(True).tag(sync=True)
    data = Unicode().tag(sync=True)
    parameters = Dict().tag(sync=True)
    voxel_paint_color = Int().tag(sync=True)

    objects = []
    COMPRESSION_LEVEL = 1

    def __init__(self, antialias=True, background_color=0xFFFFFF, camera_auto_fit=True, height=512,
                 voxel_paint_color=0):
        super(K3D, self).__init__()
        self.on_msg(self.__on_msg)

        self.__on_msg_strategy = self.__pass

        self.camera_auto_fit = camera_auto_fit
        self.parameters = {
            'antialias': antialias,
            'backgroundColor': background_color,
            'height': height,
        }
        self.voxel_paint_color = voxel_paint_color
        self.objects = []

    def __add__(self, objs):
        assert isinstance(objs, Drawable)

        if objs.set_plot(self):
            for obj in objs:
                self.objects.append(obj)
                self.data = self.__to_compressed_base64(self.__to_json(obj.data))

        return self

    def __sub__(self, objs):
        assert isinstance(objs, Drawable)

        for obj in objs:
            if obj.unset_plot(self):
                self.update(obj.id)
                self.objects.remove(obj)

        return self

    def fetch_data(self, obj):
        self.__on_msg_strategy = partial(self.__on_data, obj=obj)
        self.send(obj.id)

    def __on_msg(self, *args):
        self.__on_msg_strategy(args[1])

    def __on_data(self, data, obj):
        self.__on_msg_strategy = self.__pass
        obj.update(data)

    def update(self, obj_id, attr=None):
        data = {
            'id': obj_id,
            'attr': attr,
        }

        if attr is None:
            del data['attr']

        self.data = self.__to_compressed_base64(self.__to_json(data))

    @staticmethod
    def __to_json(data):
        return json.dumps(data, separators=(',', ':')).encode(encoding='ascii')

    @classmethod
    def __to_compressed_base64(cls, data):
        return base64.b64encode(zlib.compress(data, cls.COMPRESSION_LEVEL))

    def display(self, **kwargs):
        super(K3D, self)._ipython_display_(**kwargs)

    def __pass(self, *args, **kwargs):
        pass
