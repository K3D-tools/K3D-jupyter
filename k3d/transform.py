import numpy as np
from functools import reduce


class Transform(object):
    def __init__(self, translation=None, rotation=None, scaling=None, custom_matrix=None, parent=None):
        self.translation = translation
        self.rotation = rotation
        self.scaling = scaling
        self.parent = parent
        self.drawables = []
        self.children = []
        self.parent_matrix = parent.model_matrix if parent else np.identity(4)
        self.custom_matrix = custom_matrix if custom_matrix is not None else np.identity(4)
        self.model_matrix = np.identity(4)
        self._recompute_matrix()

    def __setattr__(self, key, value):
        is_set = hasattr(self, key)  # == False in constructor
        super(Transform, self).__setattr__(key, value)
        if is_set and key != 'model_matrix':
            self._recompute_matrix()
            self._notify_dependants()

    def _recompute_matrix(self):
        # this method shouldn't modify any fields except self.model_matrix
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
                [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c), 0.],
                [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b), 0.],
                [2*(b*d - a*c), 2*(c*d - a*b), a*a - b*b - c*c + d*d, 0.],
                [0., 0., 0., 1.]
            ])
        else:
            rotation_matrix = np.identity(4)

        if self.scaling is not None:
            scaling_matrix = np.diag(np.append(self.scaling, 1.0))
        else:
            scaling_matrix = np.identity(4)

        self.model_matrix = reduce(np.dot, [
            translation_matrix, rotation_matrix, scaling_matrix, self.custom_matrix, self.parent_matrix
        ])

    def _add_child(self, transform):
        self.children.append(transform)

    def _add_drawable(self, drawable):
        self.drawables.append(drawable)

    def _parent_updated(self):
        self.parent_matrix = self.parent.model_matrix
        self._recompute_matrix()
        self._notify_dependants()

    def _notify_dependants(self):
        for child in self.children:
            child._parent_updated()
        for drawable in self.drawables:
            drawable.model_matrix = self.model_matrix

