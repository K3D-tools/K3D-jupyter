============
Installation
============

--------------------
Installing from PyPi
--------------------

K3D-jupyter releases are available as wheel packages for macOS, Windows and Linux on PyPi_.

.. code-block:: bash

    pip install k3d

When using the package within `Jupyter Notebook`_, install and enable the ``k3d`` extension.

.. code-block:: bash

    jupyter nbextension install --py --user k3d
    jupyter nbextension enable --py --user k3d

When upgrading from an earlier version, use the following commands.

.. code-block:: bash

    pip install -U k3d
    
    jupyter nbextension install --py --user k3d
    jupyter nbextension enable --py --user k3d

---------------------
Installing with Conda
---------------------

k3d is available via the `conda-forge`_ community channel.

.. code-block:: bash
    
    conda install -c conda-forge k3d

----------------------
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

----------------------
Installing from source
----------------------

For a development installation:

.. note::
    Requires git_, `Node.js`_ and npm_.

.. code-block:: bash

    git clone https://github.com/K3D-tools/K3D-jupyter.git
    cd K3D-jupyter
    pip install -e .

An editable install does not place the JupyterLab extension in the environment, so register
the built directory in place:

.. code-block:: bash

    jupyter labextension develop --overwrite .

--------------------
JupyterLab extension
--------------------

Nothing has to be installed by hand. The wheel ships a pre-built (federated) extension and
places it in the environment, where JupyterLab and Notebook 7 discover it on their own.

.. code-block:: bash

    jupyter labextension list

should report ``k3d`` as ``enabled OK``. If it does not, or if the browser console shows
``No version of module k3d is registered``, see :doc:`frontend`.

.. note::
    ``jupyter labextension install`` belonged to JupyterLab 2 and 3, where extensions were
    built from npm sources into the application. That command no longer exists in JupyterLab 4.

.. Links
.. _PyPi: https://pypi.org/project/k3d/
.. _conda-forge: https://anaconda.org/conda-forge/k3d
.. _Jupyter Notebook: https://jupyter.org/
.. _git: https://git-scm.com/
.. _Node.js: https://nodejs.org/en/
.. _npm: https://www.npmjs.com/