.. _factory.label:

factory.label
=============

.. autofunction:: k3d.factory.label

.. seealso::
    - :ref:`factory.text`
    - :ref:`factory.text2d`

**Examples**

Basic

.. code-block:: python3

    import k3d

    plt_label1 = k3d.label('Insert text here',
                            position=(1, 1, 1))
    plt_label2 = k3d.label('Insert text here (HTML)',
                            position=(-1, -1, -1),
                            is_html=True)

    plot = k3d.plot()
    plot += plt_label1
    plot += plt_label2
    plot.display()

.. k3d_plot ::
  :filename: plots/label_basic_plot.py

Modes

.. code-block:: python3

    import k3d

    plt_points = k3d.points([[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                            point_size=0.5,
                            shader='flat',
                            colors=[0xff0000, 0x00ff00, 0x0000ff])

    plt_label_dynamic = k3d.label('Dynamic',
                                  position=(1, 1, 1),
                                  mode='dynamic',
                                  label_box=False,
                                  color=0xff0000)
    plt_label_local = k3d.label('Local',
                                position=(0, 0, 0),
                                mode='local',
                                label_box=False,
                                color=0x00ff00)
    plt_label_side = k3d.label('Side',
                                position=(-1, -1, -1),
                                mode='side',
                                label_box=False,
                                color=0x0000ff)

    plot = k3d.plot()
    plot += plt_points
    plot += plt_label_dynamic
    plot += plt_label_local
    plot += plt_label_side
    plot.display()

.. k3d_plot ::
  :filename: plots/label_modes_plot.py