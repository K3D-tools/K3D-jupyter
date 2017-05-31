import base64
import json
import warnings
import zlib

import ipywidgets as widgets
from traitlets import Unicode, Bool, Int, List

from ._version import version_info, __version__
from .factory import Factory
from .objects import Drawable

from .colormaps.paraview_color_maps import paraview_color_maps
from .colormaps.basic_color_maps import basic_color_maps
from .colormaps.matplotlib_color_maps import matplotlib_color_maps


class K3D(widgets.DOMWidget, Factory):
    version = version_info
    _view_name = Unicode('K3DView').tag(sync=True)
    _model_name = Unicode('K3DModel').tag(sync=True)
    _view_module = Unicode('k3d').tag(sync=True)
    _model_module = Unicode('k3d').tag(sync=True)

    _view_module_version = Unicode('~' + __version__).tag(sync=True)
    _model_module_version = Unicode('~' + __version__).tag(sync=True)

    objects = []
    COMPRESSION_LEVEL = 1

    data = Unicode().tag(sync=True)

    # readonly
    antialias = Bool().tag(sync=True)
    height = Int().tag(sync=True)

    # read-write
    camera_auto_fit = Bool(True).tag(sync=True)
    grid_auto_fit = Bool(True).tag(sync=True)
    grid = List().tag(sync=True)
    background_color = Int().tag(sync=True)
    voxel_paint_color = Int().tag(sync=True)
    camera = List().tag(sync=True)

    basic_color_maps = basic_color_maps
    paraview_color_maps = paraview_color_maps
    matplotlib_color_maps = matplotlib_color_maps

    def __init__(self, antialias=True, background_color=0xFFFFFF, camera_auto_fit=True, grid_auto_fit=True, height=512,
                 voxel_paint_color=0, grid=[-1, -1, -1, 1, 1, 1]):
        super(K3D, self).__init__()
        self.on_msg(self.__on_msg)

        self.antialias = antialias
        self.camera_auto_fit = camera_auto_fit
        self.grid_auto_fit = grid_auto_fit
        self.grid = grid
        self.background_color = background_color
        self.voxel_paint_color = voxel_paint_color
        self.height = height

        self.objects = []

    def __iadd__(self, objs):
        assert isinstance(objs, Drawable)

        if objs.set_plot(self):
            for obj in objs:
                self.objects.append(obj)

                sync_data = obj.data
                sync_data['k3dOperation'] = 'Insert'
                self.data = self.__to_compressed_base64(self.__to_json(sync_data))

        return self

    def __add__(self, objs):
        warnings.warn('Using plus operator to add objects to plot is discouraged in favor of +=')
        return self.__iadd__(objs)

    def __isub__(self, objs):
        assert isinstance(objs, Drawable)

        for obj in objs:
            if obj.unset_plot(self):
                self.update(obj.id)
                self.objects.remove(obj)

        return self

    def __sub__(self, objs):
        warnings.warn('Using minus operator to remove objects from plot is discouraged in favor of -=')
        return self.__isub__(objs)

    def fetch_data(self, obj):
        self.send(obj.id)

    def __on_msg(self, *args):
        json = args[1]
        if json['type'] == 'object':
            for obj in self.objects:
                if obj.id == json['id']:
                    obj.update(json['data'])

    def update(self, obj_id, attr=None):
        data = {
            'k3dOperation': 'Update',
            'id': obj_id,
            'attr': attr,
        }

        if attr is None:
            data['k3dOperation'] = 'Delete'

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
