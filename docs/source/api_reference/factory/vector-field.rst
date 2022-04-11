.. _factory.vector_field:

factory.vector_field
====================

.. autofunction:: k3d.factory.vector_field

**Examples**

Basic

.. code-block:: python3

    import k3d
    import numpy as np

    def f(x, y):
        return np.sin(y), np.sin(x)

    H = W = 10

    vectors = np.array([[f(x, y) for x in range(W)] for y in range(H)]).astype(np.float32)
    plt_vector_field = k3d.vector_field(vectors,
                                        color=0xed6a5a)

    plot = k3d.plot()
    plot += plt_vector_field
    plot.display()

.. k3d_plot ::
  :filename: plots/vector_field_basic_plot.py

Colormap

.. code-block:: python3

    import k3d
    import numpy as np
    from k3d.colormaps import matplotlib_color_maps
    from k3d.helpers import map_colors
    from numpy.linalg import norm

    p = np.linspace(-1, 1, 10)

    def f(x, y, z):
        return y * z, x * z, x * y

    vectors = np.array([[[f(x, y, z) for x in p] for y in p] for z in p]).astype(np.float32)
    norms = np.apply_along_axis(norm, 1, vectors.reshape(-1, 3))

    plt_vector_field = k3d.vector_field(vectors)

    colors = map_colors(norms, matplotlib_color_maps.Jet, [0, 1]).astype(np.uint32)
    plt_vector_field.colors = np.repeat(colors, 2)

    plot = k3d.plot()
    plot += plt_vector_field
    plot.display()

.. k3d_plot ::
  :filename: plots/vector_field_colormap_plot.py