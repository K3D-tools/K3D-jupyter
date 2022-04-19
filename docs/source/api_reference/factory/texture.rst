.. _factory.texture:

factory.texture
===============

.. autofunction:: k3d.factory.texture

**Examples**

Basic

.. code-block:: python3

    # Texture from https://opengameart.org/content/arcade-carpet-textures-arcadecarpet512png

    import k3d

    with open('arcade_carpet_512.png', 'rb') as texture:
        data = texture.read()

    plt_texture = k3d.texture(data,
                              file_format='png')

    plot = k3d.plot()
    plot += plt_texture
    plot.display()

.. k3d_plot ::
  :filename: plots/texture_basic_plot.py

Colormap

.. attention::
    `color_map` must be used along with `attribute` and `color_range` in order to work correctly.

.. code-block:: python3

    import k3d
    import numpy as np
    from k3d.colormaps import matplotlib_color_maps

    t = np.linspace(0, 1, 100)

    plt_texture = k3d.texture(color_map=matplotlib_color_maps.Jet,
                              attribute=t,
                              color_range=[0.15, 0.85])

    plot = k3d.plot()
    plot += plt_texture
    plot.display()

.. k3d_plot ::
  :filename: plots/texture_colormap_plot.py