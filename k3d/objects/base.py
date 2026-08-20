"""Base classes and utilities for K3D objects."""

import numpy as np
from traitlets import (Any, Bool, Dict, Integer, List, TraitError,
                       Unicode, Union, validate)

from .._widget import K3DModelWidget
from ..helpers import (Array, Int,
                       array_serialization_wrap, callback_serialization_wrap,
                       to_json)

EPSILON = np.finfo(np.float32).eps

SHININESS_REMOVED = (
    "shininess was removed in 3.0.0 - use roughness and metalness instead. "
    "The equivalent is roughness = sqrt(2 / (shininess + 2)), e.g. the old default 50 -> 0.196."
)


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

    def validate(self, obj, value):
        """Validate the value, handling None by converting to empty list."""
        if value is None:
            return []
        return super(ListOrArray, self).validate(obj, value)

    def validate_elements(self, obj, value):
        if self._empty_ok and len(value) == 0:
            return list(value)
        return super(ListOrArray, self).validate_elements(obj, value)


class VoxelChunk(K3DModelWidget):
    """Voxel chunk class for selective updating voxels."""

    _kind = Unicode("chunk").tag(sync=True)

    id = Int().tag(sync=True)
    voxels = Array(dtype=np.uint8).tag(sync=True, **array_serialization_wrap("voxels"))
    coord = Array(dtype=np.uint32).tag(sync=True, **array_serialization_wrap("coord"))
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

        for k in self._synced_props:
            obj[k] = to_json(k, self[k], self, self["compression_level"])

        return obj


class Drawable(K3DModelWidget):
    """
    Base class for drawable objects and groups.
    """

    _kind = Unicode("object").tag(sync=True)

    id = Integer().tag(sync=True)
    name = Unicode(default_value=None, allow_none=True).tag(sync=True)
    group = Unicode(default_value=None, allow_none=True).tag(sync=True)
    custom_data = Dict(default_value=None, allow_none=True).tag(sync=True)
    visible = TimeSeries(Bool(True)).tag(sync=True)
    compression_level = Integer().tag(sync=True)

    # Tombstone. Unknown constructor kwargs are silently swallowed by ipywidgets, so simply
    # deleting the trait would turn every existing shininess= call into a silent visual change.
    shininess = Any(default_value=None, allow_none=True)

    @validate("shininess")
    def _shininess_removed(self, proposal):
        # None passes - factories forward their tombstone parameter unconditionally
        if proposal["value"] is None:
            return None
        raise TraitError(SHININESS_REMOVED)

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

        for k in self._synced_props:
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
