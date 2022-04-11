.. _factory.vectors:

factory.vectors
===============

.. autofunction:: k3d.factory.vectors

**Examples**

Basic

.. code-block:: python3

    import k3d
    import numpy as np

    o = np.array([[0, 0, 0],
                  [2, 3, 4]]).astype(np.float32)

    v = np.array([[1, 1, 1],
                  [-2, -2, -2]]).astype(np.float32)

    plt_vectors = k3d.vectors(origins=o,
                              vectors=v,
                              colors=[0x000000, 0xde49a1,
                                      0x000000, 0x40826d])

    plot = k3d.plot()
    plot += plt_vectors
    plot.display()

.. k3d_plot ::
  :filename: plots/vectors_basic_plot.py

Labels

.. code-block:: python3

    import k3d
    import numpy as np

    o = np.array([[1, 2, 3],
                  [2, -3, 0]]).astype(np.float32)

    v = np.array([[1, 1, 1],
                  [-4, 2, 3]]).astype(np.float32)

    labels = ['(1, 1, 1)', '(2, -3, 0)']

    plt_vectors = k3d.vectors(origins=o,
                              vectors=v,
                              origin_color=0x000000,
                              head_color=0x488889,
                              line_width=0.2,
                              use_head=False,
                              labels=labels)

    plot = k3d.plot()
    plot += plt_vectors
    plot.display()

.. k3d_plot ::
  :filename: plots/vectors_labels_plot.py