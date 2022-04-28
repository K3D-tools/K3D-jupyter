Lorenz system
=============

.. code-block:: python3

    import k3d
    import numpy as np
    from k3d.colormaps import paraview_color_maps

    def lorenz(x, y, z, s=10, r=28, b=8/3):
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return x_dot, y_dot, z_dot

    dt = 0.005
    step_cnt = 10000

    xs = np.empty((step_cnt + 1), dtype=np.float32)
    ys = np.empty((step_cnt + 1), dtype=np.float32)
    zs = np.empty((step_cnt + 1), dtype=np.float32)

    xs[0], ys[0], zs[0] = (0, 1, 1.05)

    for i in range(step_cnt):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])

        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    plt_line = k3d.line(np.vstack([xs, ys, zs]).T,
                        width=0.05,
                        attribute=xs,
                        color_map=paraview_color_maps.Hue_L60)

    plot = k3d.plot(background_color=0x1e1e1e,
                    label_color=0xf0f0f0,
                    grid_visible=False,
                    menu_visibility=False)
    plot += plt_line
    plot.display()

    plot.camera = [10.1646, -36.8108, -0.4144,
                   -1.0736, 1.5642, 25.2671,
                   0.9111, -0.0730, 0.4054]

.. k3d_plot ::
  :filename: plots/lorenz_system_plot.py