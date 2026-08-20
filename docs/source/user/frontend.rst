==========================
How the frontend is loaded
==========================

A K3D plot is a widget: the Python side holds the data, the drawing is done by JavaScript in
the browser, and the two talk over the Jupyter *comm* protocol. Nothing is rendered by the
kernel itself, which is why an installation can look complete on the Python side and still
show no graphics.

Since 3.0.0 the widget is built on anywidget_. The frontend module travels **with the widget
state itself**: the kernel serves ``k3d/static/widget.mjs`` through the comm and the browser
loads it as an ES module. There is no extension directory, nothing to register with the
application, and the same mechanism works in JupyterLab, Notebook 7, Google Colab and VS Code.

The only host requirement is ipywidgets support (anywidget rides on it). If the host can show
an ``ipywidgets.IntSlider()``, it can show a K3D plot.

.. note::
    Versions before 3.0.0 delivered the frontend as a JupyterLab federated extension plus an
    nbextension for Notebook 6, with all the registration failure modes those entailed
    (``jupyter labextension list``, ``No version of module k3d is registered`` and friends).
    None of that applies any more; stale registrations from old versions are simply ignored.

-------------------------------------
Scene objects in lazy frontends
-------------------------------------

Every scene object (mesh, volume, points...) is a widget of its own, and a plot references
them by id. Some frontends - Google Colab renders each output in its own frame - materialise
widget models lazily, so the object models may not exist where the plot renders. The plot then
fetches their state from the kernel over its own comm and keeps it updated (the ``.k3d``
binary encoding is used on the wire). This is automatic; the one practical difference is that
hover/click callbacks and volume shadow maps need real object models and stay inactive in such
frontends.

-------------------------
Where the errors are
-------------------------

Frontend initialization errors do **not** appear in the server log or in the notebook. They go
to the browser console (F12).

**The cell prints only text.** You get ``Plot(antialias=3, ...)`` followed by ``Output()`` and no
drawing area. Nothing consumed the widget's mime bundle, so ``display()`` fell back to the plain
text representation of the object. Either the page has no widget support at all, or the kernel
cannot open a comm. This is not a rendering failure - the JavaScript was never asked to draw.

**The plot area appears but stays empty.** Check the browser console for errors from
``widget.mjs`` (WebGL context, GPU blacklist) and verify WebGL2 works at all, e.g. on
https://get.webgl.org/webgl2/.

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

An editable install serves the frontend straight from the checkout's ``k3d/static``. After
changing anything under ``js/src``:

.. code-block:: bash

    cd js
    npm run build

then restart the kernel and hard-refresh the page - the module is cached per widget state, so
a stale tab keeps the old code.

.. Links
.. _anywidget: https://anywidget.dev/
.. _ipywidgets: https://ipywidgets.readthedocs.io/en/latest/
