"""Utilities module."""
import itertools
import os
import zlib
from urllib.request import urlopen

import numpy as np
from traitlets import TraitError

# import logging
# from pprint import pprint, pformat
#
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# fh = logging.FileHandler('k3d.log')
# fh.setLevel(logging.INFO)
# logger.addHandler(fh)


# pylint: disable=unused-argument
# noinspection PyUnusedLocal
def array_to_json(ar, compression_level=0, force_contiguous=True):
    """Return the serialization of a numpy array.

    Args:
        ar (ndarray): A numpy array.
        compression_level (int, optional): Level of compression [-1, 9]. Defaults to 0.
        force_contiguous (bool, optional): If True, makes the array contiguous in memory. Defaults to True.

    Raises:
        ValueError: Unsupported dtype.
        Error: Bad compression level.

    Returns:
        dict: Binary data of the array with its dtype and shape.
    """
    if ar.dtype.kind not in ["u", "i", "f"]:  # ints and floats
        raise ValueError("Unsupported dtype: %s" % ar.dtype)

    if ar.dtype == np.float64:  # WebGL does not support float64
        ar = ar.astype(np.float32)
    elif ar.dtype == np.int64:  # JS does not support int64
        ar = ar.astype(np.int32)

    # make sure it's contiguous
    if force_contiguous and not ar.flags["C_CONTIGUOUS"]:
        ar = np.ascontiguousarray(ar)

    if compression_level > 0:
        return {
            "compressed_data": zlib.compress(ar.flatten(), compression_level),
            "dtype": str(ar.dtype),
            "shape": ar.shape,
        }
    else:
        return {
            "data": memoryview(ar.flatten()),
            "dtype": str(ar.dtype),
            "shape": ar.shape,
        }


# noinspection PyUnusedLocal
def json_to_array(value, obj=None):
    """Return numpy array from serialization.

    Args:
        value (dict): Binary data of an array with its dtype and shape.
        obj (dict, optional): _description_. Defaults to None.

    Returns:
        ndarray: Numpy array or None.
    """
    if value:
        if "data" in value:
            return np.frombuffer(value["data"], dtype=value["dtype"]).reshape(
                value["shape"]
            )
        else:
            return np.frombuffer(
                zlib.decompress(value["compressed_data"]), dtype=value["dtype"]
            ).reshape(value["shape"])
    return None


def to_json(name, input, obj=None, compression_level=0):
    """Return JSON object serialization."""
    if hasattr(obj, "compression_level"):
        compression_level = obj.compression_level

    if isinstance(input, dict):
        property = obj[name]
        ret = {}
        for key, value in input.items():
            ret[str(key)] = to_json(key, value, property, compression_level)

        return ret
    elif isinstance(input, np.ndarray) and input.dtype is np.dtype(object):
        return to_json(name, input.tolist(), obj, compression_level)
    elif isinstance(input, list):
        property = obj[name]
        return [
            to_json(idx, v, property, compression_level) for idx, v in enumerate(input)
        ]
    elif isinstance(input, bytes):
        return array_to_json(np.frombuffer(input, dtype=np.uint8), compression_level)
    elif isinstance(input, np.ndarray):
        return array_to_json(input, compression_level)
    else:
        return input


def from_json(input, obj=None):
    """Return JSON object deserialization."""
    if isinstance(input, dict) \
            and "dtype" in input \
            and ("data" in input or "compressed_data" in input) \
            and "shape" in input:
        return json_to_array(input, obj)
    elif isinstance(input, list):
        return [from_json(i, obj) for i in input]
    elif isinstance(input, dict):
        ret = {}
        for key, value in input.items():
            ret[key] = from_json(value, obj)

        return ret
    else:
        return input


def array_serialization_wrap(name):
    """Return a wrap of the serialization and deserialization functions for array objects."""
    return {
        "to_json": (lambda input, obj: to_json(name, input, obj)),
        "from_json": from_json,
    }


def callback_serialization_wrap(name):
    """Return a wrap of the serialization and deserialization functions for mouse actions."""
    return {
        "to_json": (lambda input, obj: obj[name] is not None),
        "from_json": from_json,
    }


def download(url):
    """Retrieve the file at url, save it locally and return its name.

    Args:
        url (str): URL.

    Raises:
        FileNotFoundError: No such file or directory.
        HTTPError: HTTP Error 403: Forbidden.
        URLError: Temporary failure in name resolution.

    Returns:
        str: File path.
    """
    basename = os.path.basename(url)

    if os.path.exists(basename):
        return basename

    with urlopen(url) as response, open(basename, "wb") as output:
        output.write(response.read())

    return basename


def minmax(arr):
    """Return the mnimum and maximum value of an array.

    Args:
        arr (array_like): Array of numbers.

    Returns:
        list: Array of two numbers.
    """
    return [float(np.nanmin(arr)), float(np.nanmax(arr))]


def check_attribute_color_range(attribute, color_range=()):
    """Provide and return color range versus provided attribute.

    Args:
        attribute (list): Array of numbers.
        color_range (tuple, optional): Tuple of two numbers. Defaults to ().

    Returns:
        tuple: Color range.
    """
    if type(attribute) is dict or attribute.size == 0 or len(color_range) == 2:
        return color_range

    color_range = minmax(attribute)

    if color_range[0] == color_range[1]:
        color_range[1] += 1.0

    return color_range


def map_colors(attribute, color_map, color_range=()):
    """Return color mapping according to an attribute and a color map.

    The attribute represents the data on which the colormap will be apply.
    The color range allows to constraint the colormap between two values.

    Args:
        attribute (ndarray): Array of numbers.
        color_map (array_like): Array of numbers.
        color_range (tuple, optional): Tuple of two numbers. Defaults to ().

    Returns:
        ndarray: Color mapping.
    """
    a_min, a_max = check_attribute_color_range(attribute, color_range)
    map_array = np.asarray(color_map)
    map_array = map_array.reshape((map_array.size // 4, 4))

    # normalizing attribute for range lookup
    attribute = (attribute - a_min) / (a_max - a_min)

    red, green, blue = [
        np.array(
            255 * np.interp(attribute,
                            xp=map_array[:, 0], fp=map_array[:, i + 1]),
            dtype=np.int32)
        for i in range(3)
    ]

    colors = (red << 16) + (green << 8) + blue
    return colors


def bounding_corners(bounds, z_bounds=(0, 1)):
    """Return corner point coordinates for bounds array.

    z_bounds assigns Z points coordinates if bounds contains less than 5 items.

    Args:
        bounds (array_like): Array of numbers.
        z_bounds (tuple, optional): Tuple of two numbers. Defaults to (0, 1).

    Returns:
        ndarray: Corner point coordinates.
    """
    return np.array(
        list(itertools.product(bounds[:2],
             bounds[2:4], bounds[4:] or z_bounds))
    )


def min_bounding_dimension(bounds):
    """Return a minimal dimension along axis in a bounds array.

    *bounds* must be in the form *[min_x, max_x, min_y, max_y, min_z, max_z]*.

    Args:
        bounds (array_like): Array of numbers.

    Returns:
        number: Minimum value of the array.
    """
    return min(abs(x1 - x0) for x0, x1 in zip(bounds, bounds[1:]))


def shape_validation(*dimensions):
    """Create a validator callback ensuring array shape.

    Raises:
        TraitError: Expected an array of shape _ and got _.

    Returns:
        function: Shape validator function.
    """
    def validator(trait, value):
        if np.shape(value) != dimensions:
            raise TraitError(
                "Expected an array of shape %s and got %s" % (
                    dimensions, value.shape)
            )

        return value

    return validator


def sparse_voxels_validation():
    """Check sparse voxels for array shape and values.

    Raises:
        TraitError: Expected an array of shape (N, 4) and got _.
        TraitError: Voxel coordinates and values must be non-negative.

    Returns:
        function: Sparse voxels validator function.
    """
    def validator(trait, value):
        if len(value.shape) != 2 or value.shape[1] != 4:
            raise TraitError(
                "Expected an array of shape (N, 4) and got %s" % (value.shape,)
            )

        if (value.astype(np.int16) < 0).any():
            raise TraitError(
                "Voxel coordinates and values must be non-negative")

        return value

    return validator


def quad(w, h):
    """Return the vertices and indices of a w * h quadrilateral.

    Args:
        w (number): Quadrilateral width.
        h (number): Quadrilateral height.

    Returns:
        tuple (ndarray, ndarray): Array of vertices and indices.
    """
    w /= 2
    h /= 2

    vertices = np.array([-w, -h, -0, w, -h, 0, w, h, 0, -w, h, 0],
                        dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    return vertices, indices


def get_bounding_box(model_matrix, boundary=[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]):
    """Return the boundaries of the model_matrix.

    Args:
        model_matrix (ndarray): Matrix of numbers. Must have four columns.

        boundary (list, optional): Array of numbers. Defaults to [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5].
            Must be of the form `[min_x, max_x, min_y, max_y, min_z, max_z]`.

    Returns:
        ndarray: Model matrix boundaries.
    """
    b_min = np.array([boundary[0], boundary[2], boundary[4], 0])
    b_max = np.array([boundary[1], boundary[3], boundary[5], 0])

    b_min = model_matrix.dot(b_min)
    b_max = model_matrix.dot(b_max)

    return np.dstack([b_min[0:3], b_max[0:3]]).flatten()


def get_bounding_box_points(arr, model_matrix):
    """Return the minimum and maximum coordinates on x, y, z axes.

    Args:
        arr (ndarray): Array of vertices. `[x, y, z]`.
        model_matrix (ndarray): Matrix of numbers. Must have four columns.

    Returns:
        ndarray: Array of numbers. `[min_x, max_x, min_y, max_y, min_z, max_z]`.
    """
    d = arr.flatten()

    if d.shape[0] < 3:
        d = np.array([0, 0, 0])

    # fmt: off
    boundary = np.array([
        np.min(d[0::3]), np.max(d[0::3]),
        np.min(d[1::3]), np.max(d[1::3]),
        np.min(d[2::3]), np.max(d[2::3]),
    ])
    # fmt: on

    return get_bounding_box(model_matrix, boundary)


def get_bounding_box_point(position):
    """Return the boundaries of a position.

    Args:
        position (array_like): Array of numbers.

    Returns:
        ndarray: Array of numbers.
    """
    return np.dstack([np.array(position), np.array(position)]).flatten()
