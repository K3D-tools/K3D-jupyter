import numpy as np
from ipython_genutils.py3compat import string_types, PY3


def get_dimensions(shape, *dimensions):
    return dimensions if len(shape) < len(dimensions) else [val or shape[i] for i, val in enumerate(dimensions)]


def validate_vectors_size(length, vector_size):
    expected_vectors_size = 2 if length is None else 3

    if vector_size is not expected_vectors_size:
        raise TypeError('Invalid vectors size: expected %d, %d given' % (expected_vectors_size, vector_size))


def get_base_matrix(xmin, xmax, ymin, ymax, zmin, zmax):
    for name, value in locals().items():
        try:
            float(value)
        except (TypeError, ValueError):
            raise TypeError('%s: expected float, %s given' % (name, type(value).__name__))

    matrix = np.diagflat(np.array((xmax - xmin, ymax - ymin, zmax - zmin, 1.0), np.float32, order='C'))
    matrix[0:3, 3] = ((xmax + xmin) / 2.0, (ymax + ymin) / 2.0, (zmax + zmin) / 2.0)

    return matrix


def get_model_matrix(model_matrix, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5):
    model_matrix = np.array(model_matrix, np.float32).reshape(4, -1)

    if model_matrix.shape != (4, 4):
        raise ValueError(
            'model_matrix: expected 4x4 matrix, %s given' % 'x'.join(str(i) for i in model_matrix.shape))

    return np.dot(get_base_matrix(xmin, xmax, ymin, ymax, zmin, zmax), model_matrix)


# from ipyvolume

def array_to_json(ar, obj=None):
    return ar.tolist() if ar is not None else None


def array_to_binary(ar, obj=None, force_contiguous=True):
    if ar.dtype.kind not in ['u', 'i', 'f']:  # ints and floats
        raise ValueError("unsupported dtype: %s" % (ar.dtype))
    if ar.dtype == np.float64:  # WebGL does not support float64, case it here
        ar = ar.astype(np.float32)
    if ar.dtype == np.int64:  # JS does not support int64
        ar = ar.astype(np.int32)
    if force_contiguous and not ar.flags["C_CONTIGUOUS"]:  # make sure it's contiguous
        ar = np.ascontiguousarray(ar)
    return {'buffer': memoryview(ar), 'dtype': str(ar.dtype), 'shape': ar.shape}


def array_to_binary_or_json(ar, obj=None):
    if ar is None:
        return None
    element = ar
    dimension = 0
    try:
        while True:
            element = element[0]
            dimension += 1
    except:
        pass
    try:
        element = element.item()  # for instance get back the value from array(1)
    except:
        pass
    if isinstance(element, string_types):
        return array_to_json(ar)

    return array_to_binary(ar)


def from_json_to_array(value, obj=None):
    return np.frombuffer(value['buffer'], dtype=value['dtype']) if value else None


array_serialization = dict(to_json=array_to_binary_or_json, from_json=from_json_to_array)
