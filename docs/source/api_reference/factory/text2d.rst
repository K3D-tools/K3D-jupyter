.. _factory.text2d:

factory.text2d
==============

.. autofunction:: k3d.factory.text2d

.. seealso::
    - :ref:`factory.label`
    - :ref:`factory.text`
    - :ref:`factory.text`

**Examples**

Basic

.. code-block:: python3

    import k3d

    plt_text1 = k3d.text2d('Insert text here',
                          position=(0, 0))
    plt_text2 = k3d.text2d('Insert text here (HTML)',
                          position=(0, 0.1),
                          is_html=True)

    plot = k3d.plot()
    plot += plt_text1
    plot += plt_text2
    plot.display()

.. k3d_plot ::
  :filename: plots/text2d_basic_plot.py

Positions

.. code-block:: python3

    import k3d

    plt_text_0_0 = k3d.text2d('(0,0)',
                              position=(0, 0),
                              reference_point='lt')
    plt_text_0_05 = k3d.text2d('(0,0.5)',
                              position=(0, 0.5),
                              reference_point='lc')
    plt_text_0_1 = k3d.text2d('(0,1)',
                              position=(0, 1),
                              reference_point='lb')
    plt_text_05_0 = k3d.text2d('(0.5,0)',
                              position=(0.5, 0),
                              reference_point='ct')
    plt_text_1_0 = k3d.text2d('(1,0)',
                              position=(1, 0),
                              reference_point='rt')
    plt_text_1_05 = k3d.text2d('(1,0.5)',
                              position=(1, 0.5),
                              reference_point='rc')
    plt_text_1_1 = k3d.text2d('(1,1)',
                              position=(1, 1),
                              reference_point='rb')
    plt_text_05_1 = k3d.text2d('(0.5,1)',
                              position=(0.5, 1),
                              reference_point='cb')
    plt_text_05_05 = k3d.text2d('(0.5,0.5)',
                                position=(0.5, 0.5),
                                reference_point='cc')

    plot = k3d.plot()
    plot += plt_text_0_0
    plot += plt_text_0_05
    plot += plt_text_0_1
    plot += plt_text_05_0
    plot += plt_text_1_0
    plot += plt_text_1_05
    plot += plt_text_1_1
    plot += plt_text_05_1
    plot += plt_text_05_05
    plot.display()

.. k3d_plot ::
  :filename: plots/text2d_positions_plot.py