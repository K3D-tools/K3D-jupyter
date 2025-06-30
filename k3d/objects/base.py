"""Base classes and utilities for K3D objects."""

import ipywidgets as widgets
import numpy as np
import warnings
from traitlets import (
    Any,
    Bool,
    Bytes,
    Dict,
    Float,
    Int,
    Integer,
    List,
    TraitError,
    Unicode,
    Union,
    validate,
)
from traittypes import Array

from .._version import __version__ as version
from ..helpers import (
    array_serialization_wrap,
    callback_serialization_wrap,
    get_bounding_box_point,
    get_bounding_box_points,
    get_bounding_box,
    to_json,
)

EPSILON = np.finfo(np.float32).eps


class TimeSeries(Union):
    def __init__(self, trait):
        if isinstance(trait, list):
            Union.__init__(self, trait + [Dict(t) for t in trait])
        else:
            Union.__init__(self, [trait, Dict(trait)])


class SingleOrList(Union):
    def __init__(self, trait):
        Union.__init__(self, [trait, List(trait)])


class ListOrArray(List):
    _cast_types = (tuple, np.ndarray)

    def __init__(self, *args, **kwargs):
        self._empty_ok = kwargs.pop("empty_ok", False)
        List.__init__(self, *args, **kwargs)

    def validate_elements(self, obj, value):
        if self._empty_ok and len(value) == 0:
            return list(value)
        return super(ListOrArray, self).validate_elements(obj, value)


class VoxelChunk(widgets.Widget):
    """Voxel chunk class for selective updating voxels."""
    _model_name = Unicode("ChunkModel").tag(sync=True)
    _model_module = Unicode("k3d").tag(sync=True)
    _model_module_version = Unicode(version).tag(sync=True)

    id = Int().tag(sync=True)
    voxels = Array(dtype=np.uint8).tag(
        sync=True, **array_serialization_wrap("voxels"))
    coord = Array(dtype=np.uint32).tag(
        sync=True, **array_serialization_wrap("coord"))
    multiple = Int().tag(sync=True)
    compression_level = Integer().tag(sync=True)

    def push_data(self, field):
        self.notify_change({"name": field, "type": "change"})

    def __init__(self, **kwargs):
        self.id = id(self)
        super(VoxelChunk, self).__init__(**kwargs)

    def __getitem__(self, name):
        return getattr(self, name)

    def get_binary(self):
        obj = {}

        for k, v in self.traits().items():
            if "sync" in v.metadata:
                obj[k] = to_json(k, self[k], self, self["compression_level"])

        return obj


class Drawable(widgets.Widget):
    """
    Base class for drawable objects and groups.
    """

    _model_name = Unicode("ObjectModel").tag(sync=True)
    _model_module = Unicode("k3d").tag(sync=True)
    _model_module_version = Unicode(version).tag(sync=True)

    id = Integer().tag(sync=True)
    name = Unicode(default_value=None, allow_none=True).tag(sync=True)
    group = Unicode(default_value=None, allow_none=True).tag(sync=True)
    custom_data = Dict(default_value=None, allow_none=True).tag(sync=True)
    visible = TimeSeries(Bool(True)).tag(sync=True)
    compression_level = Integer().tag(sync=True)

    def __getitem__(self, name):
        return getattr(self, name)

    def __init__(self, **kwargs):
        self.id = id(self)

        super(Drawable, self).__init__(**kwargs)

    def __iter__(self):
        return (self,).__iter__()

    def __add__(self, other):
        return Group(self, other)

    def fetch_data(self, field):
        """Request updating the value of a field modified in browser.

        For data modified in the widget on the browser side, this triggers an asynchronous
        update of the value in the Python kernel.

        Only specific features require this mechanism, e.g. the in-browser editing of voxels.

        Arguments:
            field: `str`.
                The field name."""
        self.send({"msg_type": "fetch", "field": field})

    def push_data(self, field):
        """Request updating the value of a field modified in backend.

        For data modified in the backend side, this triggers an asynchronous
        update of the value in the browser widget.

        Only specific features require this mechanism, e.g. the in-browser editing of voxels.

        Arguments:
            field: `str`.
                The field name."""
        self.notify_change({"name": field, "type": "change"})

    def _ipython_display_(self, **kwargs):
        """Called when `IPython.display.display` is called on the widget."""
        import k3d

        plot = k3d.plot()
        plot += self
        plot.display()

    def clone(self):
        from .utils import clone_object
        return clone_object(self)

    def get_binary(self):
        obj = {}

        for k, v in self.traits().items():
            if "sync" in v.metadata:
                obj[k] = to_json(k, self[k], self, self["compression_level"])

        return obj


class DrawableWithVoxelCallback(Drawable):
    """
    Base class for drawable with voxels callback handling
    """

    click_callback = None
    hover_callback = None

    def __init__(self, **kwargs):
        super(DrawableWithVoxelCallback, self).__init__(**kwargs)

        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, content, buffers):
        if content.get("msg_type", "") == "click_callback":
            if self.click_callback is not None:
                self.click_callback(
                    content["coord"]["x"], content["coord"]["y"], content["coord"]["z"]
                )

        if content.get("msg_type", "") == "hover_callback":
            if self.hover_callback is not None:
                self.hover_callback(
                    content["coord"]["x"], content["coord"]["y"], content["coord"]["z"]
                )


class DrawableWithCallback(Drawable):
    """
    Base class for drawable with callback handling
    """

    click_callback = Any(default_value=None, allow_none=True).tag(
        sync=True, **callback_serialization_wrap("click_callback")
    )
    hover_callback = Any(default_value=None, allow_none=True).tag(
        sync=True, **callback_serialization_wrap("hover_callback")
    )

    def __init__(self, **kwargs):
        super(DrawableWithCallback, self).__init__(**kwargs)

        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, content, buffers):
        if content.get("msg_type", "") == "click_callback":
            if self.click_callback is not None:
                self.click_callback(content)

        if content.get("msg_type", "") == "hover_callback":
            if self.hover_callback is not None:
                self.hover_callback(content)


class Group(Drawable):
    """
    An aggregated group of Drawables, itself a Drawable.

    It can be inserted or removed from a Plot including all members.
    """

    __objs = None

    def __init__(self, *args):
        self.__objs = tuple(
            self.__assert_drawable(drawable)
            for drawables in args
            for drawable in drawables
        )

    def __iter__(self):
        return self.__objs.__iter__()

    def __setattr__(self, key, value):
        """Special method override which allows for setting model matrix for all members of the group."""
        if key == "model_matrix":
            for d in self:
                d.model_matrix = value
        else:
            super(Group, self).__setattr__(key, value)

    @staticmethod
    def __assert_drawable(arg):
        assert isinstance(arg, Drawable)

        return arg 