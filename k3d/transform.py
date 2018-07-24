import numpy as np
from functools import reduce
import weakref

_epsilon = 1e-6


def get_bounds_fit_matrix(xmin, xmax, ymin, ymax, zmin, zmax):
    """Create a 4x4 transform matrix which maps the default bounding box ([-0.5, 0.5] in all dimensions) into
    a custom bounding box ([xmin, xmax, ymin, ymax, zmin, zmax]).

    It is used for fitting fields (VectorField, Surface, MachingCubes) into custom domains."""

    for name, value in locals().items():
        try:
            float(value)
        except (TypeError, ValueError):
            raise TypeError('%s: expected float, %s given' % (name, type(value).__name__))

    matrix = np.diagflat(np.array((xmax - xmin, ymax - ymin, zmax - zmin, 1.0), np.float32, order='C'))
    matrix[0:3, 3] = ((xmax + xmin) / 2.0, (ymax + ymin) / 2.0, (zmax + zmin) / 2.0)

    return matrix


class Transform(object):
    """
    Abstraction of a 4x4 model transformation matrix with hierarchy support.
    """

    def __init__(self, bounds=None, translation=None, rotation=None, scaling=None, custom_matrix=None, parent=None):
        """
        Transform constructor.

        :param bounds: List[float] (xmin, xmax, ymin, ymax, zmin, zmax)
        :param translation: List[float] (dx, dy, dz) - translation vector
        :param rotation: List[float] (gamma, axis_x, axis_y, axis_z) - angle in radians, then rotation axis vector
        :param scaling: List[float] (s_x, s_y, s_z) - 3 scaling coefficients
        :param custom_matrix: np.array - 4x4 arbitrary transform matrix
        :param parent: `Transform` optional parent transform, which is applied before this transform
        """

        self.bounds = bounds
        self.translation = translation
        self.rotation = rotation
        self.scaling = scaling
        self.parent = parent
        if parent is not None:
            parent._add_child(self)
        self.drawables = []
        self.children = []
        self.parent_matrix = parent.model_matrix if parent else np.identity(4, dtype=np.float32)
        self.custom_matrix = custom_matrix if custom_matrix is not None else np.identity(4, dtype=np.float32)
        self.model_matrix = np.identity(4, dtype=np.float32)
        self._recompute_matrix()

    def __setattr__(self, key, value):
        """Set attributes with conversion to ndarray where needed."""
        is_set = hasattr(self, key)  # == False in constructor

        # parameter canonicalization and some validation via reshaping
        if value is None:
            # TODO: maybe forbid for some fields
            pass
        elif key == 'translation':
            value = np.array(value, dtype=np.float32).reshape(3, 1)
        elif key == 'rotation':
            value = np.array(value, dtype=np.float32).reshape(4)
            value[0] = np.cos(value[0] / 2)

            norm = np.linalg.norm(value[1:4])
            needed_norm = np.sqrt(1 - value[0] * value[0])
            if abs(norm - needed_norm) > _epsilon:
                if norm < _epsilon:
                    raise ValueError('Norm of (x, y, z) part of quaternion too close to zero')
                value[1:4] = value[1:4] / norm * needed_norm
            # assert abs(np.linalg.norm(value) - 1.0) < _epsilon
        elif key == 'scaling':
            value = np.array(value, dtype=np.float32).reshape(3)
        elif key in ['parent_matrix', 'custom_matrix', 'model_matrix']:
            value = np.array(value, dtype=np.float32).reshape((4, 4))

        super(Transform, self).__setattr__(key, value)

        if is_set and key != 'model_matrix':
            self._recompute_matrix()
            self._notify_dependants()

    def __repr__(self):
        return 'Transform(bounds={!r}, translation={!r}, rotation={!r}, scaling={!r})'.format(
            self.bounds, self.translation, self.rotation, self.scaling
        )

    def _recompute_matrix(self):
        # this method shouldn't modify any fields except self.model_matrix

        if self.bounds is None or len(self.bounds) == 0:
            fit_matrix = np.identity(4)
        else:
            if len(self.bounds) == 6:
                xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            elif len(self.bounds) == 4:
                xmin, xmax, ymin, ymax = self.bounds
                zmin, zmax = -0.5, 0.5
            elif len(self.bounds) == 2:
                # 1-D bounds for a 2D strip - why not?
                xmin, xmax = self.bounds
                ymin, ymax, zmin, zmax = -0.5, 0.5, -0.5, 0.5
            else:
                raise ValueError('Wrong size of bounds array ({}), should be 4 for 2D or 6 for 3D bounds.'.format(
                    self.bounds
                ))
            fit_matrix = get_bounds_fit_matrix(xmin, xmax, ymin, ymax, zmin, zmax)

        if self.translation is not None:
            translation_matrix = np.vstack((
                np.hstack((np.identity(3), np.array(self.translation).reshape(3, 1))),
                np.array([0., 0., 0., 1.]).reshape(1, 4)
            ))
        else:
            translation_matrix = np.identity(4)

        if self.rotation is not None:
            a, b, c, d = self.rotation
            rotation_matrix = np.array([
                [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c), 0.],
                [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b), 0.],
                [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d, 0.],
                [0., 0., 0., 1.]
            ])
        else:
            rotation_matrix = np.identity(4)

        if self.scaling is not None:
            scaling_matrix = np.diag(np.append(self.scaling, 1.0))
        else:
            scaling_matrix = np.identity(4)

        self.model_matrix = reduce(np.dot, [
            translation_matrix, rotation_matrix, scaling_matrix, fit_matrix, self.custom_matrix, self.parent_matrix
        ])

    def _add_child(self, transform):
        self.children.append(weakref.ref(transform))

    def add_drawable(self, drawable):
        """Register a Drawable to have its model_matrix overwritten after changes to the transform or its parent."""
        self.drawables.append(weakref.ref(drawable))

    def parent_updated(self):
        """Read updated parent transform matrix and update own model_matrix.

        This method should be normally only called by parent Transform to notify its children.
        """
        if self.parent is not None:
            self.parent_matrix = self.parent.model_matrix
        self._recompute_matrix()
        self._notify_dependants()

    def _notify_dependants(self):
        extant_children = []
        for child_ref in self.children:
            child = child_ref()
            if child is not None:
                child.parent_updated()
                extant_children.append(child_ref)
        self.children[:] = extant_children

        extant_drawables = []
        for drawable_ref in self.drawables:
            drawable = drawable_ref()
            if drawable is not None:
                drawable.model_matrix = self.model_matrix
                extant_drawables.append(drawable_ref)
        self.drawables[:] = extant_drawables


def process_transform_arguments(drawable, **kwargs):
    """Process keyword arguments dictionary for a drawable to create a Transform for it.

    Keyword arguments:
        transform: `Transform`. An existing transform object to be (re-)used for the drawable.
            This is useful if the transform should have a parent.
        xmin: `float`. Lower bound in the X dimension for array fields of scalars or vectors.
        xmax: `float`. Upper bound in the X dimension for array fields of scalars or vectors.
        ymin: `float`. Lower bound in the Y dimension for array fields of scalars or vectors.
        ymax: `float`. Upper bound in the Y dimension for array fields of scalars or vectors.
        zmin: `float`. Lower bound in the Z dimension for array fields of scalars or vectors.
        zmax: `float`. Upper bound in the Z dimension for array fields of scalars or vectors.
        bounds: `array_like`. Array of bounds (bounding box or rectangle) for array fields of scalars or vectors.
            This array can be of size 6 (xmin, xmax, ymin, ymax, zmin, zmax) for 3D fields,
            or of size 4 (xmin, xmax, ymin, ymax) for 2D fields.
            Takes precedence over separate xmin, xmax, ... etc. arguments.
        translation=kwargs.get('translation'),
        rotation=kwargs.get('rotation'),
        scaling=kwargs.get('scaling'),
        model_matrix
    """

    if 'transform' in kwargs:
        transform = kwargs['transform']
        if not isinstance(transform, Transform):
            raise ValueError('Provided transform argument is not a Transform object')
    else:
        separate_bounds = [
            kwargs.get('xmin', -0.5),
            kwargs.get('xmax', 0.5),
            kwargs.get('ymin', -0.5),
            kwargs.get('ymax', 0.5),
            kwargs.get('zmin', -0.5),
            kwargs.get('zmax', 0.5)
        ]

        transform_kwargs = dict(
            bounds=kwargs.get('bounds', separate_bounds),
            translation=kwargs.get('translation'),
            rotation=kwargs.get('rotation'),
            scaling=kwargs.get('scaling'),
            custom_matrix=kwargs.get('model_matrix')
        )
        transform = Transform(**transform_kwargs)

    transform.add_drawable(drawable)
    transform.parent_updated()
    drawable.transform = transform

    return drawable
