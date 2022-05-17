.. _stl:

stl
===

.. autofunction:: k3d.factory.stl

.. seealso::
    - :ref:`vtk_poly_data`

Examples
--------

:download:`skull_w_jaw.stl <./assets/skull_w_jaw.stl>`

.. code-block:: python3

    # Model from https://www.thingiverse.com/thing:819046/

    import k3d

    with open('skull_w_jaw.stl', 'rb') as stl:
        data = stl.read()

    plt_skull = k3d.stl(data, color=0xe3dac9)

    plot = k3d.plot()
    plot += plt_skull
    plot.display()

.. k3d_plot ::
  :filename: plots/stl_basic_plot.py