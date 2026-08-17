============
Installation
============

--------------------
Installing from PyPi
--------------------

K3D-jupyter releases are available as wheel packages for macOS, Windows and Linux on PyPi_.

.. code-block:: bash

    pip install k3d

That is the whole installation. Since 2.19.0 the widget is built on anywidget_ and ships its
own frontend module inside the wheel - there is no extension to install or enable, and the
same package works in JupyterLab, Jupyter Notebook, Google Colab and VS Code notebooks.

When upgrading from an earlier version:

.. code-block:: bash

    pip install -U k3d

.. note::
    Versions before 2.19.0 required ``jupyter nbextension install/enable`` (Notebook 6) or a
    prebuilt labextension. Those steps are obsolete - if old registrations linger, they are
    ignored by the new widget.

---------------------
Installing with Conda
---------------------

k3d is available via the `conda-forge`_ community channel.

.. code-block:: bash

    conda install -c conda-forge k3d

------------
Google Colab
------------

.. code-block:: bash

    !pip install k3d

Nothing else is needed. The previous ritual (``output.enable_custom_widget_manager()`` and
``k3d.switch_to_text_protocol()``) belongs to versions before 2.19.0.

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

The build step compiles the frontend into ``k3d/static`` (``widget.mjs`` for the widget,
``standalone.js`` for snapshots and headless use). After editing anything under ``js/src``,
rebuild and hard-refresh the browser:

.. code-block:: bash

    cd js
    npm run build

No ``jupyter labextension develop`` step exists any more - the kernel serves the frontend
module straight from ``k3d/static``.

.. Links
.. _PyPi: https://pypi.org/project/k3d/
.. _conda-forge: https://anaconda.org/conda-forge/k3d
.. _anywidget: https://anywidget.dev/
.. _git: https://git-scm.com/
.. _Node.js: https://nodejs.org/en/
.. _npm: https://www.npmjs.com/
