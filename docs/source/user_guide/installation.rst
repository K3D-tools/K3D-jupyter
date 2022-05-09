.. _installation:

Installation
============

Installing an official release
------------------------------

k3d releases are available as wheel packages for macOS, Windows and Linux on PyPi_.
Install it using ``pip``:

.. code-block:: bash

    pip install k3d

When using the package within `Jupyter Notebook`_, install and enable the ``k3d`` extension:

.. code-block:: bash

    jupyter nbextension install --py --user k3d
    jupyter nbextension enable --py --user k3d

When upgrading from an earlier version, use the following commands:

.. code-block:: bash

    pip install -U k3d
    jupyter nbextension install --py --user k3d
    jupyter nbextension enable --py --user k3d

Conda package
-------------

k3d is available via the `conda-forge`_ community channel:

.. code-block:: bash

    conda install -c conda-forge k3d

Installing from GitHub
----------------------

You can install directly from the `repository <https://github.com/K3D-tools/K3D-jupyter>`_:

.. note::
    Requires git_, `Node.js`_ and npm_.

.. code-block:: bash

    pip install git+https://github.com/K3D-tools/K3D-jupyter

You can also install the most up-to-date development version:

.. code-block:: bash

    pip install git+https://github.com/K3D-tools/K3D-jupyter@devel

If you want to install any historical version, replace ``devel`` with any tag or commit hash.

Installing from source
----------------------

For a development installation:

.. note::
    Requires git_, `Node.js`_ and npm_.

.. code-block:: bash

    git clone https://github.com/K3D-tools/K3D-jupyter.git
    cd K3D-jupyter
    pip install -e .

Then, if required, JupyterLab installation:

.. code-block:: bash

    jupyter labextension install ./js

JupyterLab extension
--------------------

If required, you can install the JupyterLab extension:

.. note::
    Do not run inside the K3D-Jupyter directory.

.. code-block:: bash

    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    jupyter labextension install k3d

.. important::
  Please notice that support for JupyterLab is still experimental.

.. Links
.. _PyPi: https://pypi.org/project/k3d/
.. _conda-forge: https://anaconda.org/conda-forge/k3d
.. _Jupyter Notebook: https://jupyter.org/
.. _git: https://git-scm.com/
.. _Node.js: https://nodejs.org/en/
.. _npm: https://www.npmjs.com/