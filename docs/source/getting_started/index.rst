Installation quick-start
========================

Install using `pip`_:

.. code-block:: bash

    pip install k3d


Install using `conda`_:

.. code-block:: bash

  conda install -c conda-forge k3d

.. seealso::
    - :ref:`installation`

First plot
==========

.. code-block:: python3

    import k3d

    plot = k3d.line([[0, 0, 0],
                     [1, 1, 1]])

    plot.display()

.. k3d_plot::
  :filename: first_plot.py


.. Links
.. _pip: https://pypi.org/project/k3d/
.. _conda: https://anaconda.org/conda-forge/k3d