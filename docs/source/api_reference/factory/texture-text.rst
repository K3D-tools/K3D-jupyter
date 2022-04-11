.. _factory.texture_text:

factory.texture_text
====================

.. autofunction:: k3d.factory.texture_text

**Examples**

Basic

.. code-block:: python3

    import k3d

    plt_texture_text = k3d.texture_text('Texture',
                                        position=[0, 0, 0],
                                        font_face='Calibri',
                                        font_weight=600,
                                        color=0xa2ffc8)

    plot = k3d.plot()
    plot += plt_texture_text
    plot.display()

.. k3d_plot ::
  :filename: plots/texture_text_basic_plot.py