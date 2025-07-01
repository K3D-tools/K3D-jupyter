"""Utilities module."""
import base64
import itertools
import msgpack
import numpy as np
import os
import zlib
from traitlets import TraitError
from urllib.request import urlopen

from ._protocol import get_protocol
from typing import Union, List, Optional, Dict, Any, Tuple


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

    Parameters
    ----------
    ar : ndarray
        A numpy array.
    compression_level : int, optional
        Level of compression [-1, 9], by default 0.
    force_contiguous : bool, optional
        Make the array contiguous in memory, by default True.

    Returns
    -------
    dict
        Binary data of the array with its dtype and shape.

    Raises
    ------
    ValueError
        Unsupported dtype.
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
        ret = {
            "compressed_data": zlib.compress(ar.flatten(), compression_level),
            "dtype": str(ar.dtype),
            "shape": ar.shape,
        }
    else:
        ret = {
            "data": memoryview(ar.flatten()),
            "dtype": str(ar.dtype),
            "shape": ar.shape,
        }

    if get_protocol() == 'text':
        return 'base64_' + base64.b64encode(msgpack.packb(ret, use_bin_type=True)).decode('ascii')
    else:
        return ret


# noinspection PyUnusedLocal
def json_to_array(value, obj=None):
    """Return numpy array from serialization.

    Parameters
    ----------
    value : dict
        Binary data of an array with its dtype and shape.
    obj : dict, optional
        Object, by default None.

    Returns
    -------
    ndarray
        Numpy array or None.
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
    elif isinstance(input, np.number):
        return input.tolist()
    else:
        return input


def from_json(input, obj=None):
    """Return JSON object deserialization."""
    if isinstance(input, str) and input[0:7] == 'base64_':
        input = msgpack.unpackb(base64.b64decode(input[7:]))

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

    Parameters
    ----------
    url : str
        URL.

    Returns
    -------
    str
        File path.
    """
    basename = os.path.basename(url)

    if os.path.exists(basename):
        return basename

    with urlopen(url) as response, open(basename, "wb") as output:
        output.write(response.read())

    return basename


def minmax(arr):
    """Return the minimum and maximum value of an array.

    Parameters
    ----------
    arr : array_like
        Array of numbers.

    Returns
    -------
    list
        Array of two numbers.
    """
    return [float(np.nanmin(arr)), float(np.nanmax(arr))]


def check_attribute_color_range(attribute: Union[np.ndarray, Dict[str, np.ndarray]], color_range: Union[List[float], Tuple[float, ...]] = ()) -> List[float]:
    """Return color range versus provided attribute.

    Parameters
    ----------
    attribute : list or dict (for timeseries)
        Array of numbers.
    color_range : tuple, optional
        Two numbers, by default ().

    Returns
    -------
    tuple
        Color range.
    """

    if color_range is None:
        color_range = []
    
    if len(color_range) == 2:
        return color_range
    elif type(attribute) is dict:
        t = [minmax(attribute[k]) for k in attribute.keys()]
        color_range = [min([v[0] for v in t]), max([v[1] for v in t])]
    elif attribute.size == 0:
        return color_range
    else:
        color_range = minmax(attribute)

    if color_range[0] == color_range[1]:
        color_range[1] += 1.0

    return color_range


def map_colors(attribute: np.ndarray, color_map: Union[List[List[float]], np.ndarray], color_range: Union[List[float], Tuple[float, ...]] = ()) -> np.ndarray:
    """Return color mapping according to an attribute and a colormap.

    The attribute represents the data on which the colormap will be applied.
    The color range allows to constraint the colormap between two values.

    Parameters
    ----------
    attribute : ndarray
        Array of numbers.
    color_map : array_like
        Array of numbers.
    color_range : tuple, optional
        Two numbers, by default ().

    Returns
    -------
    ndarray
        Color mapping.
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


def bounding_corners(bounds: Union[List[float], np.ndarray], z_bounds: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """Return corner point coordinates for bounds array.

    `z_bounds` assigns Z points coordinates if bounds contains less than 5 items.

    Parameters
    ----------
    bounds : array_like
        Array of numbers.
    z_bounds : tuple, optional
        Two numbers, by default (0, 1).

    Returns
    -------
    ndarray
        Corner points coordinates.
    """
    return np.array(
        list(itertools.product(bounds[:2],
                               bounds[2:4], bounds[4:] or z_bounds))
    )


def min_bounding_dimension(bounds: Union[List[float], np.ndarray]) -> float:
    """Return the minimal dimension along axis in a bounds array.

    `bounds` must be of the form [min_x, max_x, min_y, max_y, min_z, max_z].

    Parameters
    ----------
    bounds : array_like
        Array of numbers.

    Returns
    -------
    number
        Minimum value of the array.
    """
    return min(abs(x1 - x0) for x0, x1 in zip(bounds, bounds[1:]))


def shape_validation(*dimensions):
    """Create a validator callback ensuring array shape.

    Returns
    -------
    function
        Shape validator function.

    Raises
    ------
    TraitError
        Expected an array of shape _ and got _.
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

    Returns
    -------
    function
        Sparse voxels validator function.

    Raises
    ------
    TraitError
        Expected an array of shape (N, 4) and got _.
    TraitError
        Voxel coordinates and values must be non-negative.
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


def quad(w: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return the vertices and indices of a `w` * `h` quadrilateral.

    Parameters
    ----------
    w : number
        Quadrilateral width.
    h : number
        Quadrilateral height.

    Returns
    -------
    tuple
        Array of vertices and indices.
    """
    w /= 2
    h /= 2

    vertices = np.array([-w, -h, -0, w, -h, 0, w, h, 0, -w, h, 0],
                        dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    return vertices, indices


def get_bounding_box(model_matrix, boundary=[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]):
    """Return the boundaries of a model matrix.

    Parameters
    ----------
    model_matrix : ndarray
        Matrix of numbers. Must have four columns.
    boundary : list, optional
        Array of numbers, by default [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5].
        Must be of the form [min_x, max_x, min_y, max_y, min_z, max_z].

    Returns
    -------
    ndarray
        Model matrix boundaries.
    """
    b_min = np.array([boundary[0], boundary[2], boundary[4], 0])
    b_max = np.array([boundary[1], boundary[3], boundary[5], 0])

    b_min = model_matrix.dot(b_min)
    b_max = model_matrix.dot(b_max)

    return np.dstack([b_min[0:3], b_max[0:3]]).flatten()


def get_bounding_box_points(arr, model_matrix):
    """Return the minimum and maximum coordinates on x, y, z axes.

    Parameters
    ----------
    arr : ndarray
        Array of vertices [x, y, z].
    model_matrix : ndarray
        Matrix of numbers. Must have four columns.

    Returns
    -------
    ndarray
        Array of numbers [min_x, max_x, min_y, max_y, min_z, max_z].
    """
    d = arr.flatten()

    if d.shape[0] < 3:
        d = np.array([0, 0, 0])

    # fmt: off
    boundary = np.array([
        np.nanmin(d[0::3]), np.nanmax(d[0::3]),
        np.nanmin(d[1::3]), np.nanmax(d[1::3]),
        np.nanmin(d[2::3]), np.nanmax(d[2::3]),
    ])
    # fmt: on

    return get_bounding_box(model_matrix, boundary)


def get_bounding_box_point(position):
    """Return the boundaries of a position.

    Parameters
    ----------
    position : array_like
        Array of numbers.

    Returns
    -------
    ndarray
        Array of numbers.
    """
    return np.dstack([np.array(position), np.array(position)]).flatten()


def unify_color_map(cm):
    cm[0::4] = (cm[0::4] - np.min(cm[0::4])) / (np.max(cm[0::4]) - np.min(cm[0::4]))
    return cm


def contour(data, bounds, values, clustering_factor=0):
    import pyvista
    grid = pyvista.ImageData(dimensions=data.shape[::-1],
                             spacing=(bounds[1::2] - bounds[::2]) / np.array(data.shape[::-1]),
                             origin=bounds[::2])

    grid.point_data["values"] = data.flatten()
    mesh = grid.contour(values, grid.point_data["values"], method='flying_edges')

    mesh.compute_normals(inplace=True)

    if clustering_factor > 1:
        import pyacvd

        clus = pyacvd.Clustering(mesh)
        clus.cluster(mesh.n_points // clustering_factor)
        return clus.create_mesh()
    else:
        return mesh
