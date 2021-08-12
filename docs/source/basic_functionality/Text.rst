Text
====

There are four kinds of text objects:

 - on the 3D scene `k3d.text`, renders in canvas,
 - on the 2D canvas `k3d.text2d`, renders in canvas,
 - `k3d.texture_text` renders as texture in WebGL,
 - `k3d.label` text in canvas, with connector to a point on the scene.

K3D-jupyter uses KaTeX for mathematical notation as a default. One can force plain
html text using `is_html=True` option.

.. code::

    import k3d
    import numpy as np

    g = 9.81
    v0 = 24

    plot = k3d.plot()

    for alpha_deg in [10, 30, 45, 60, 85]:
        alpha = np.radians(alpha_deg)
        t_end = 2 * v0 * np.sin(alpha) / g
        t = np.linspace(0, t_end, 100)

        # note .T at the end, k3d takes data (x1,y1,z1),(x2,y2,z2)...
        traj3d = np.stack([v0 * t * np.cos(alpha), \
                           20 * alpha + np.zeros_like(t), \
                           v0 * t * np.sin(alpha) - g * t ** 2 / 2]).T.astype(np.float32)

        plt_traj = k3d.line(traj3d)
        plt_text = k3d.text('h_{max}',
                            position=[float(np.cos(alpha) * t_end * v0 / 2),
                                      float(20 * alpha),
                                      float((v0 * np.sin(alpha)) ** 2 / (2 * g))],
                            color=0xff0000, size=1)
        plt_text2d = k3d.text2d(
            r'\text{ballistic trajectory: }\; h=v_0 t \sin \alpha - \frac{g t^2}{2}',
            position=[0.0, 0.0], color=0x0000ff, size=1)
        plt_texture_text = k3d.texture_text('START', position=[0, 0, 0],
                                            font_face='Calibri', color=255, size=5)

        if alpha_deg == 45:
            plt_label = k3d.label('Optimal angle', traj3d[-1].tolist(), mode='dynamic',
                                  is_html=True, color=0xff00f0)
            plot += plt_label

        plot += plt_text
        plot += plt_text2d
        plot += plt_texture_text
        plot += plt_traj

    plot.display()

.. k3d_plot ::
   :filename: text/Text01.py
