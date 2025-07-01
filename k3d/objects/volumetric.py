"""Volumetric objects for K3D."""

import numpy as np
import warnings
from traitlets import (
    Bool,
    Float,
    Int,
    List,
    Unicode,
    validate,
)
from traittypes import Array

from ..helpers import (
    array_serialization_wrap,
    get_bounding_box,
    get_bounding_box_points,
    shape_validation,
    sparse_voxels_validation,
)
from .base import (
    Drawable,
    DrawableWithCallback,
    DrawableWithVoxelCallback,
    TimeSeries,
    ListOrArray,
)


class MarchingCubes(DrawableWithCallback):
    """
    An isosurface in a scalar field obtained through Marching Cubes algorithm.

    The default domain of the scalar field is -0.5 < x, y, z < 0.5.
    If the domain should be different, the bounding box needs to be transformed using the model_matrix.

    Attributes:
        scalar_field: `array_like`.
            A 3D scalar field of values.
        level: `float`.
            Value at the computed isosurface.
        spacings_x: `array_like`.
            A spacings in x axis. Should match to scalar_field shape.
        spacings_y: `array_like`.
            A spacings in y axis. Should match to scalar_field shape.
        spacings_z: `array_like`.
            A spacings in z axis. Should match to scalar_field shape.
        color: `int`.
            Packed RGB color of the isosurface (0xff0000 is red, 0xff is blue).
        attribute: `array_like`.
            Array of float attribute for the color mapping, coresponding to each vertex.
        color_map: `list`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        opacity_function: `array`.
            A list of float tuples (attribute value, opacity), sorted by attribute value. The first
            tuples should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        flat_shading: `bool`.
            Whether mesh should display with flat shading.
        shininess: `float`.
            Shininess of object material.
        opacity: `float`.
            Opacity of mesh.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    scalar_field = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("scalar_field")
    )
    spacings_x = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("spacings_x")
    )
    spacings_y = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("spacings_y")
    )
    spacings_z = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("spacings_z")
    )
    level = Float().tag(sync=True)
    attribute = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("attribute")
    )
    color_map = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    color_range = TimeSeries(ListOrArray(minlen=2, maxlen=2, empty_ok=True)).tag(
        sync=True
    )
    opacity_function = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("opacity_function")
    )
    color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    wireframe = Bool().tag(sync=True)
    flat_shading = Bool().tag(sync=True)
    shininess = TimeSeries(Float(default_value=50.0)).tag(sync=True)
    opacity = TimeSeries(
        Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)

    def __init__(self, **kwargs):
        super(MarchingCubes, self).__init__(**kwargs)

        self.set_trait("type", "MarchingCubes")


class VolumeSlice(DrawableWithCallback):
    """Create a Volume slice drawable.

    Arguments:
        volume: `array_like`.
            3D array of `float`
        color_map: `list`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        opacity_function: `array`.
            A list of float tuples (attribute value, opacity), sorted by attribute value. The first
            typles should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
        opacity: `float`.
            Opacity of slice.
        slice_x: `int`.
            Number of slice. -1 for hidden.
        slice_y: `int`.
            Number of slice. -1 for hidden.
        slice_z: `int`.
            Number of slice. -1 for hidden.
        interpolation: `int`.
            0 - no interpolation, 1 - linear, 2 - cubic.
        mask: `array_like`.
            3D array of `int` in range (0, 255).
        active_masks: `array_like`.
            List of values from mask.
        color_map_masks: `list`.
            Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).
            The color defined at index i is for voxel value (i+1), e.g.:
        mask_opacity: `Float`.
            Mask enhanced coefficient.
        name: `string`.
            A name of a object
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix.
        model_matrix: `array_like`.
            4x4 model transform matrix."""

    type = Unicode(read_only=True).tag(sync=True)
    volume = TimeSeries(Array()).tag(sync=True, **array_serialization_wrap('volume'))
    color_map = TimeSeries(Array(dtype=np.float32)).tag(sync=True,
                                                        **array_serialization_wrap('color_map'))
    color_range = TimeSeries(ListOrArray(minlen=2, empty_ok=True)).tag(sync=True)
    opacity_function = TimeSeries(Array(dtype=np.float32)).tag(sync=True,
                                                               **array_serialization_wrap(
                                                                   'opacity_function'))
    opacity = TimeSeries(Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    slice_x = TimeSeries(Int()).tag(sync=True)
    slice_y = TimeSeries(Int()).tag(sync=True)
    slice_z = TimeSeries(Int()).tag(sync=True)
    interpolation = TimeSeries(Int()).tag(sync=True)
    mask = Array(dtype=np.uint8).tag(sync=True, **array_serialization_wrap('mask'))
    active_masks = Array(dtype=np.uint8).tag(sync=True, **array_serialization_wrap('active_masks'))
    color_map_masks = Array(dtype=np.uint32).tag(sync=True,
                                                 **array_serialization_wrap('color_map_masks'))
    mask_opacity = TimeSeries(Float()).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(sync=True, **array_serialization_wrap(
        'model_matrix'))

    def __init__(self, **kwargs):
        super(VolumeSlice, self).__init__(**kwargs)

        self.set_trait('type', 'VolumeSlice')

    @validate('volume')
    def _validate_volume(self, proposal):
        if type(proposal['value']) is dict:
            return proposal['value']

        if type(proposal['value']) is list:
            return proposal['value']

        if type(proposal['value']) is np.ndarray and proposal['value'].dtype is np.dtype(object):
            return proposal['value'].tolist()

        if proposal['value'].shape == (0,):
            return np.array(proposal['value'], dtype=np.float32)

        required = [np.float16, np.float32]
        actual = proposal['value'].dtype

        if actual not in required:
            warnings.warn('wrong dtype: %s (%s required)' % (actual, required))

            return proposal['value'].astype(np.float32)

        return proposal['value']

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)


class Volume(Drawable):
    """
    3D volumetric data.

    By default, the volume are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).

    Attributes:
        volume: `array_like`.
            3D array of `float`.
        color_map: `array_like`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        opacity_function: `array`.
            A list of float tuples (attribute value, opacity), sorted by attribute value. The first
            typles should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        samples: `float`.
            Number of iteration per 1 unit of space.
        alpha_coef: `float`.
            Alpha multiplier.
        shadow: `str`.
            Type of shadow on volume.

            Legal values are:

            :`off`: shadow disabled,

            :`on_demand`: update shadow map on demand ( self.shadow_map_update() ),

            :`dynamic`: update shadow map automaticaly every shadow_delay.
        shadow_delay: `float`.
            Minimum number of miliseconds between shadow map updates.
        shadow_res: `int`.
            Resolution of shadow map.
        interpolation: `bool`.
            Whether volume raycasting should interpolate data or not.
        mask: `array_like`.
            3D array of `int` in range (0, 255).
        mask_opacities: `array_like`.
            List of opacity values for mask.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    volume = TimeSeries(Array()).tag(
        sync=True, **array_serialization_wrap("volume"))
    color_map = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    opacity_function = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("opacity_function")
    )
    color_range = TimeSeries(ListOrArray(minlen=2, maxlen=2, empty_ok=True)).tag(
        sync=True
    )
    samples = TimeSeries(Float()).tag(sync=True)
    alpha_coef = TimeSeries(Float()).tag(sync=True)
    gradient_step = TimeSeries(Float()).tag(sync=True)
    shadow = TimeSeries(Unicode()).tag(sync=True)
    shadow_res = TimeSeries(
        Int(min=31, max=513, default_value=128)).tag(sync=True)
    shadow_delay = TimeSeries(Float()).tag(sync=True)
    ray_samples_count = TimeSeries(
        Int(min=1, max=128, default_value=16)).tag(sync=True)
    focal_length = TimeSeries(Float()).tag(sync=True)
    focal_plane = TimeSeries(Float()).tag(sync=True)
    interpolation = TimeSeries(Bool()).tag(sync=True)
    mask = Array(dtype=np.uint8).tag(sync=True, **array_serialization_wrap('mask'))
    mask_opacities = TimeSeries(Array(dtype=np.float32)).tag(sync=True, **array_serialization_wrap(
        'mask_opacities'))
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(Volume, self).__init__(**kwargs)

        self.set_trait("type", "Volume")

    @validate("volume")
    def _validate_volume(self, proposal):
        if type(proposal["value"]) is dict:
            return proposal["value"]

        if type(proposal["value"]) is np.ndarray and proposal[
            "value"
        ].dtype is np.dtype(object):
            return proposal["value"].tolist()

        required = [np.float16, np.float32]
        actual = proposal["value"].dtype

        if actual not in required:
            warnings.warn("wrong dtype: %s (%s required)" % (actual, required))

            return proposal["value"].astype(np.float32)

        return proposal["value"]

    def shadow_map_update(self, direction=None):
        """Request updating the shadow map in browser."""

        self.send({"msg_type": "shadow_map_update", "direction": direction})

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)


class MIP(Drawable):
    """
    3D volumetric data.

    By default, the volume are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).

    Attributes:
        volume: `array_like`.
            3D array of `float`.
        color_map: `array_like`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        opacity_function: `array`.
            A list of float tuples (attribute value, opacity), sorted by attribute value. The first
            typles should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        samples: `float`.
            Number of iteration per 1 unit of space.
        gradient_step: `float`
            Gradient light step.
        mask: `array_like`.
            3D array of `int` in range (0, 255).
        mask_opacities: `array_like`.
            List of opacity values for mask.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    volume = TimeSeries(Array()).tag(
        sync=True, **array_serialization_wrap("volume"))
    color_map = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    opacity_function = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("opacity_function")
    )
    color_range = TimeSeries(ListOrArray(minlen=2, maxlen=2, empty_ok=True)).tag(
        sync=True
    )
    gradient_step = TimeSeries(Float()).tag(sync=True)
    samples = TimeSeries(Float()).tag(sync=True)
    interpolation = TimeSeries(Bool()).tag(sync=True)
    mask = Array(dtype=np.uint8).tag(sync=True, **array_serialization_wrap('mask'))
    mask_opacities = TimeSeries(Array(dtype=np.float32)).tag(sync=True, **array_serialization_wrap(
        'mask_opacities'))
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(MIP, self).__init__(**kwargs)

        self.set_trait("type", "MIP")

    @validate("volume")
    def _validate_volume(self, proposal):
        if type(proposal["value"]) is dict:
            return proposal["value"]

        if type(proposal["value"]) is np.ndarray and proposal[
            "value"
        ].dtype is np.dtype(object):
            return proposal["value"].tolist()

        required = [np.float16, np.float32]
        actual = proposal["value"].dtype

        if actual not in required:
            warnings.warn("wrong dtype: %s (%s required)" % (actual, required))

            return proposal["value"].astype(np.float32)

        return proposal["value"]

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)


class Voxels(DrawableWithVoxelCallback):
    """
    3D volumetric data.

    Different grid size, shape and rotation can be obtained using model_matrix.

    Attributes:
        voxels: `array_like`.
            3D array of `int` in range (0, 255).
            0 means empty voxel, 1 and above refer to consecutive color_map entries.
        color_map: `array_like`.
            Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).

            The color defined at index i is for voxel value (i+1), e.g.:

           | color_map = [0xff, 0x00ff]
           | voxels =
           | [
           | 0, # empty voxel
           | 1, # blue voxel
           | 2  # red voxel
           | ]

        model_matrix: `array_like`.
            4x4 model transform matrix.
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        opacity: `float`.
            Opacity of voxels.
        outlines: `bool`.
            Whether mesh should display with outlines.
        outlines_color: `int`.
            Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue)
    """

    type = Unicode(read_only=True).tag(sync=True)
    voxels = Array(dtype=np.uint8).tag(
        sync=True, **array_serialization_wrap("voxels"))
    color_map = Array(dtype=np.uint32).tag(
        sync=True, **array_serialization_wrap("voxels")
    )
    wireframe = Bool().tag(sync=True)
    outlines = Bool().tag(sync=True)
    outlines_color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    opacity = TimeSeries(
        Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(Voxels, self).__init__(**kwargs)

        self.set_trait("type", "Voxels")

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)


class SparseVoxels(DrawableWithVoxelCallback):
    """
    3D volumetric data.

    By default, the voxels are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using the model_matrix.

    Attributes:
        sparse_voxels: `array_like`.
            2D array of `coords` in format [[x,y,z,v],[x,y,z,v]].
            v = 0 means empty voxel, 1 and above refer to consecutive color_map entries.
        space_size: `array_like`.
            Width, Height, Length of space
        color_map: `array_like`.
            Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).
        model_matrix: `array_like`.
            4x4 model transform matrix.
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        opacity: `float`.
            Opacity of voxels.
        outlines: `bool`.
            Whether mesh should display with outlines.
        outlines_color: `int`.
            Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue)
    """

    type = Unicode(read_only=True).tag(sync=True)
    sparse_voxels = (
        Array(dtype=np.uint16)
        .tag(sync=True, **array_serialization_wrap("sparse_voxels"))
        .valid(sparse_voxels_validation())
    )
    space_size = (
        Array(dtype=np.uint32)
        .tag(sync=True, **array_serialization_wrap("space_size"))
        .valid(shape_validation(3))
    )
    color_map = Array(dtype=np.uint32).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    wireframe = Bool().tag(sync=True)
    outlines = Bool().tag(sync=True)
    outlines_color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    opacity = TimeSeries(
        Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(SparseVoxels, self).__init__(**kwargs)

        self.set_trait("type", "SparseVoxels")

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)


class VoxelsGroup(DrawableWithVoxelCallback):
    """
    3D volumetric data.

    By default, the voxels are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using the model_matrix.

    Attributes:
        voxels_group: `array_like`.
            List of `chunks` in format {voxels: np.array, coord: [x,y,z], multiple: number}.
        space_size: `array_like`.
            Width, Height, Length of space
        color_map: `array_like`.
            Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).
        model_matrix: `array_like`.
            4x4 model transform matrix.
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        opacity: `float`.
            Opacity of voxels.
        outlines: `bool`.
            Whether mesh should display with outlines.
        outlines_color: `int`.
            Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue)
    """

    type = Unicode(read_only=True).tag(sync=True)

    _hold_remeshing = Bool(default_value=False).tag(sync=True)

    voxels_group = List(default_value=[]).tag(sync=True, **array_serialization_wrap("voxels_group"))
    chunks_ids = List(default_value=[]).tag(sync=True)

    space_size = Array(dtype=np.uint32).tag(
        sync=True, **array_serialization_wrap("space_size")
    )
    color_map = Array(dtype=np.uint32).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    wireframe = Bool().tag(sync=True)
    outlines = Bool().tag(sync=True)
    outlines_color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    opacity = TimeSeries(
        Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(VoxelsGroup, self).__init__(**kwargs)

        self.set_trait("type", "VoxelsGroup")

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)
