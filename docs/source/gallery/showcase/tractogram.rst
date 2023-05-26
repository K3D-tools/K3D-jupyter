Tractogram
==========

.. admonition:: References

    - :ref:`plot`
    - :ref:`lines`

:download:`tractogram.npz <./assets/tractogram.npz>`

.. code-block:: python3

    filepath = os.path.join('tractogram.npz')

    data = np.load(filepath)['data']

    v = data.copy()
    v[1:] = (v[1:] - v[:-1])
    v = np.absolute((v / np.linalg.norm(v, axis=1)[..., np.newaxis]))
    v = (v * 255).astype(np.int32)
    colors = np.sum((v * np.array([1, 256, 256 * 256])), axis=1).astype(np.uint32)

    streamlines = k3d.line(data, shader='simple', colors=colors)

    plot = k3d.plot(grid_visible=False,
                    camera_auto_fit=False,
                    background_color=0,
                    screenshot_scale=1.0)
    plot += streamlines

    plot.camera = [-50.0, 125.0, 40.0,
                   -1.0, 0.5, -5.0,
                   0.0, -0.25, 1.0]
    plot.display()

.. k3d_plot ::
  :filename: plots/tractogram_plot.py