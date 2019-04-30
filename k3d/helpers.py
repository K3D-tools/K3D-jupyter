"""
Utilities module.
"""

import os
import numpy as np
import itertools
import zlib

# import logging
#
# from pprint import pprint, pformat
#
# logger = logging.getLogger("k3d")
# fh = logging.FileHandler('k3d.log')
# logger.addHandler(fh)
# logger.setLevel(logging.DEBUG)


# pylint: disable=unused-argument
# noinspection PyUnusedLocal
def array_to_binary(ar, obj=None, force_contiguous=True):
    """Pre-process numpy array for serialization in traittypes.Array."""
    if ar.dtype.kind not in ['u', 'i', 'f']:  # ints and floats
        raise ValueError("unsupported dtype: %s" % ar.dtype)

    if ar.dtype == np.float64:  # WebGL does not support float64
        ar = ar.astype(np.float32)
    elif ar.dtype == np.int64:  # JS does not support int64
        ar = ar.astype(np.int32)

    if force_contiguous and not ar.flags["C_CONTIGUOUS"]:  # make sure it's contiguous
        ar = np.ascontiguousarray(ar)

    if obj.compression_level > 0:
        return {'compressed_buffer': zlib.compress(ar, obj.compression_level), 'dtype': str(ar.dtype),
                'shape': ar.shape}
    else:
        return {'buffer': memoryview(ar), 'dtype': str(ar.dtype), 'shape': ar.shape}


# noinspection PyUnusedLocal
def from_json_to_array(value, obj=None):
    """Post-process traittypes.Array after deserialization to numpy array."""
    if value:
        return np.frombuffer(value['buffer'], dtype=value['dtype']).reshape(value['shape'])
    return None


def to_json(input, obj=None, force_contiguous=True):
    if isinstance(input, dict):
        ret = {}
        for key, value in input.items():
            ret[key] = to_json(value, obj, force_contiguous)

        return ret
    elif isinstance(input, list):
        return [to_json(i, obj) for i in input]
    elif isinstance(input, np.ndarray):
        return array_to_binary(input, obj, force_contiguous)
    else:
        return input


def from_json(input, obj=None):
    # logger.info('from_json:' + pformat(input))

    if isinstance(input, dict) and 'dtype' in input and 'buffer' in input and 'shape' in input:
        return from_json_to_array(input, obj)
    elif isinstance(input, list):
        return [from_json(i, obj) for i in input]
    elif isinstance(input, dict):
        ret = {}
        for key, value in input.items():
            ret[key] = from_json(value, obj)

        return ret
    else:
        return input


array_serialization = dict(to_json=to_json, from_json=from_json)


def download(url):
    """Retrieve the file at url, save it locally and return the path."""
    basename = os.path.basename(url)
    if os.path.exists(basename):
        return basename

    print('Downloading: {}'.format(basename))

    # 2/3 compatibility hacks
    try:
        from urllib.request import urlopen
    except ImportError:
        import urllib2
        import contextlib
        urlopen = lambda url_: contextlib.closing(urllib2.urlopen(url_))

    with urlopen(url) as response, open(basename, 'wb') as output:
        output.write(response.read())

    return basename


def minmax(iterable):
    """Return [min(iterable), max(iterable)].

    This should be a built in function in Python, and has even been proposed on Python-ideas newsgroup.
    This is not to be confused with the algorithm for finding winning strategies in 2-player games."""
    return [np.nanmin(iterable), np.nanmax(iterable)]


def check_attribute_range(attribute, color_range=()):
    """Provide color range versus provided attribute, compute color range if necessary.

    If the attribute is empty or color_range has 2 elements, returns color_range unchanged.
    Computes color range as [min(attribute), max(attribute)].
    When min(attribute) == max(attribute) returns [min(attribute), min(attribute)+1]."""
    if attribute.size == 0 or len(color_range) == 2:
        return color_range
    color_range = minmax(attribute)
    if color_range[0] == color_range[1]:
        color_range[1] += 1.0
    return color_range


def map_colors(attribute, color_map, color_range=()):
    a_min, a_max = check_attribute_range(attribute, color_range)
    map_array = np.asarray(color_map)
    map_array = map_array.reshape((map_array.size // 4, 4))
    attribute = (attribute - a_min) / (a_max - a_min)  # normalizing attribute for range lookup
    red, green, blue = [np.array(255 * np.interp(attribute, xp=map_array[:, 0], fp=map_array[:, i + 1]), dtype=np.int32)
                        for i in range(3)]
    colors = (red << 16) + (green << 8) + blue
    return colors


def bounding_corners(bounds, z_bounds=(0., 1)):
    """Return corner point coordinates for bounds array."""
    return np.array(list(itertools.product(bounds[:2], bounds[2:4], bounds[4:] or z_bounds)))


def min_bounding_dimension(bounds):
    """Return a minimal dimension along axis in a bounds ([min_x, max_x, min_y, max_y, min_z, max_z]) array."""
    return min(abs(x1 - x0) for x0, x1 in zip(bounds, bounds[1:]))
