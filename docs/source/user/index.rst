===============
Getting started
===============

----------
Quickstart
----------

Install using `pip`_:

.. code-block:: bash

    pip install k3d


Install using `conda`_:

.. code-block:: bash

    conda install -c conda-forge k3d

----------
First plot
----------

.. code-block:: python3

    import k3d

    plot = k3d.plot()
    plot += k3d.line([[0, 0, 0],
                     [1, 1, 1]])

    plot.display()

.. k3d_plot::
    :filename: plots/first_plot.py

.. _pip: https://pypi.org/project/k3d/
.. _conda: https://anaconda.org/conda-forge/k3d

.. toctree::
    :maxdepth: 1
    :hidden:

    why
    install
    usage
