import ipywidgets as widgets
from traitlets import Unicode, Bool, Int, List
import warnings

from ._version import __version__
from .objects import Drawable


class Plot(widgets.DOMWidget):
    """
    Main K3D widget.
    """

    _view_name = Unicode('PlotView').tag(sync=True)
    _model_name = Unicode('PlotModel').tag(sync=True)
    _view_module = Unicode('k3d').tag(sync=True)
    _model_module = Unicode('k3d').tag(sync=True)

    _view_module_version = Unicode('~' + __version__).tag(sync=True)
    _model_module_version = Unicode('~' + __version__).tag(sync=True)

    # readonly
    antialias = Bool().tag(sync=True)
    height = Int().tag(sync=True)
    object_ids = List().tag(sync=True)

    # read-write
    camera_auto_fit = Bool(True).tag(sync=True)
    grid_auto_fit = Bool(True).tag(sync=True)
    grid = List().tag(sync=True)
    background_color = Int().tag(sync=True)
    voxel_paint_color = Int().tag(sync=True)
    camera = List().tag(sync=True)
    screenshot = Unicode().tag(sync=True)

    objects = []

    def __init__(self, antialias=True, background_color=0xFFFFFF, camera_auto_fit=True, grid_auto_fit=True, height=512,
                 voxel_paint_color=0, grid=[-1, -1, -1, 1, 1, 1]):
        super(Plot, self).__init__()

        self.antialias = antialias
        self.camera_auto_fit = camera_auto_fit
        self.grid_auto_fit = grid_auto_fit
        self.grid = grid
        self.background_color = background_color
        self.voxel_paint_color = voxel_paint_color
        self.height = height

        self.object_ids = []
        self.objects = []

    def __iadd__(self, objs):
        assert isinstance(objs, Drawable)

        for obj in objs:
            if obj.id not in self.object_ids:
                self.object_ids = self.object_ids + [obj.id]
                self.objects.append(obj)

        return self

    def __add__(self, objs):
        warnings.warn('Using plus operator to add objects to plot is discouraged in favor of +=')
        return self.__iadd__(objs)

    def __isub__(self, objs):
        assert isinstance(objs, Drawable)

        for obj in objs:
            self.object_ids = [id for id in self.object_ids if id != obj.id]
            if obj in self.objects:
                self.objects.remove(obj)

        return self

    def __sub__(self, objs):
        warnings.warn('Using minus operator to remove objects from plot is discouraged in favor of -=')
        return self.__isub__(objs)

    def display(self, **kwargs):
        super(Plot, self)._ipython_display_(**kwargs)

    def fetch_screenshot(self, callback=None):
        def default_callback():
            if callback:
                callback()
            else:
                print('Done fetching screenshot')
            self.unobserve_all('screenshot')

        self.observe('screenshot', default_callback)
        self.send({'msg_type': 'fetch_screenshot'})
