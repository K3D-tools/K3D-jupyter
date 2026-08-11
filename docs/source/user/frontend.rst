==========================
How the frontend is loaded
==========================

A K3D plot is an ipywidgets_ widget: the Python side holds the data, the drawing is done by
JavaScript in the browser, and the two talk over the Jupyter *comm* protocol. Nothing is
rendered by the kernel itself, which is why an installation can look complete on the Python
side and still show no graphics.

Two things have to be in place: a **kernel** that implements comms, and a **frontend
extension** registered with whichever Jupyter application is serving the page.

-------------------------
The two delivery paths
-------------------------

The JavaScript reaches the browser in one of two ways, depending on the application. They are
independent - having one does not give you the other.

**JupyterLab, and Notebook 7 which is built on it** load a *federated* (pre-built) extension.
It is discovered from the environment, not from the notebook:

.. code-block:: bash

    <sys.prefix>/share/jupyter/labextensions/k3d/

The wheel installs it there. That directory's ``package.json`` carries the entry point under
``jupyterlab._build.load``, pointing at ``static/remoteEntry.<hash>.js``; the application reads
it when the page is served, so a **browser refresh** is enough to pick up a newly registered
extension - the server does not need restarting.

**Notebook 6 and earlier** load an nbextension, which has to be installed and enabled
explicitly:

.. code-block:: bash

    jupyter nbextension install --py --user k3d
    jupyter nbextension enable --py --user k3d

.. note::
    ``jupyter labextension install`` is a JupyterLab 2/3 command that rebuilt the application
    from npm sources. It no longer exists in JupyterLab 4 and is not how K3D is installed.

To see what is actually registered:

.. code-block:: bash

    jupyter labextension list
    jupyter nbextension list

A working JupyterLab installation reports ``k3d <version> enabled OK``.

-------------------------
Where the errors are
-------------------------

Frontend initialization errors do **not** appear in the server log or in the notebook. They go
to the browser console (F12). Two signatures cover almost every report:

**A model class fails to load.** The console reports ``Failed to load model class 'PlotModel'
from module 'k3d'``, usually together with ``No version of module k3d is registered``. The
Python side is fine and the widget was created, but the browser has no K3D extension to build it
with. Check the delivery path above for the application you are using.

**The cell prints only text.** You get ``Plot(antialias=3, ...)`` followed by ``Output()`` and no
drawing area. Nothing consumed the widget's mime bundle, so ``display()`` fell back to the plain
text representation of the object. Either the page has no widget support at all, or the kernel
cannot open a comm. This is not a rendering failure - the JavaScript was never asked to draw.

------------------------------
Kernels other than ipykernel
------------------------------

K3D uses no kernel API of its own: it neither opens comms nor imports ``ipykernel``, and leaves
all of that to ipywidgets. A kernel therefore has to support ipywidgets before it can show a
plot, and K3D can do nothing about a kernel that does not.

If a plot shows only its text representation under a custom kernel, check ipywidgets on its own
first:

.. code-block:: python3

    import ipywidgets

    ipywidgets.IntSlider()

If that slider does not appear either, the problem is below K3D - the kernel is not carrying
comm messages to the frontend - and it has to be solved there.

-----------------------------------
Running from a source checkout
-----------------------------------

An editable install does not place the labextension in the environment, so JupyterLab will not
find it. Register the built directory in place:

.. code-block:: bash

    jupyter labextension develop --overwrite .

This symlinks ``k3d/labextension`` into ``share/jupyter/labextensions/k3d``. It is a per
environment operation: in a container built fresh each run it has to be repeated, or baked into
the image.

.. Links
.. _ipywidgets: https://ipywidgets.readthedocs.io/en/latest/
