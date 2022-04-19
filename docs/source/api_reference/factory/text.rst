.. _factory.text:

factory.text
============

.. autofunction:: k3d.factory.text

.. seealso::
    - :ref:`factory.label`
    - :ref:`factory.text2d`
    - :ref:`factory.texture_text`

**Examples**

Basic

.. code-block:: python3

    import k3d

    plt_text1 = k3d.text('Insert text here',
                         position=(1, 1, 1))
    plt_text2 = k3d.text('Insert text here (HTML)',
                         position=(-1, -1, -1),
                         is_html=True)

    plot = k3d.plot()
    plot += plt_text1
    plot += plt_text2
    plot.display()

.. k3d_plot ::
  :filename: plots/text_basic_plot.py

Reference points

.. code-block:: python3

    import k3d

    plt_square = k3d.mesh(vertices=[[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]],
                          indices=[[0, 1, 2], [2, 1, 0], [0, 2, 3], [3, 2, 0]],
                          colors=[0xff0000, 0x00ff00, 0x0000ff, 0xffff00])

    plt_text_lt = k3d.text('Left-Top',
                          position=(-1, 0, 1), reference_point='lt',
                          color=0xffff00, size=0.7)
    plt_text_lb = k3d.text('Left-Bottom',
                          position=(-1, 0, -1), reference_point='lb',
                          color=0x0000ff, size=0.7)
    plt_text_lc = k3d.text('Left-Center',
                          position=(-1, 0, 0), reference_point='lc',
                          color=0x808080, size=0.7)
    plt_text_rt = k3d.text('Right-Top',
                          position=(1, 0, 1), reference_point='rt',
                          color=0xff0000, size=0.7)
    plt_text_rb = k3d.text('Right-Bottom',
                          position=(1, 0, -1), reference_point='rb',
                          color=0x00ff00, size=0.7)
    plt_text_rc = k3d.text('Right-Center',
                          position=(1, 0, 0), reference_point='rc',
                          color=0x808000, size=0.7)
    plt_text_ct = k3d.text('Center-Top',
                          position=(0, 0, 1), reference_point='ct',
                          color=0xff8000, size=0.7)
    plt_text_cb = k3d.text('Center-Bottom',
                          position=(0, 0, -1), reference_point='cb',
                          color=0x008080, size=0.7)
    plt_text_cc = k3d.text('Center-Center',
                          position=(0, 0, 0), reference_point='cc',
                          color=0xff00ff, size=0.7)

    plot = k3d.plot()
    plot += plt_square
    plot += plt_text_lt
    plot += plt_text_lb
    plot += plt_text_lc
    plot += plt_text_rt
    plot += plt_text_rb
    plot += plt_text_rc
    plot += plt_text_ct
    plot += plt_text_cb
    plot += plt_text_cc
    plot.display()

.. k3d_plot ::
  :filename: plots/text_reference_points_plot.py