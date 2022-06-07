Orbits
======

.. admonition:: References

    - :ref:`line`
    - :ref:`paraview_color_maps`
    - :ref:`plot`
    - :ref:`points`
    - :ref:`time_series`

.. note::
    Because this example relies on randomness, the live version configuration
    is hard-coded to ensure the same results for everyone.

.. code-block:: python3

    import k3d
    import numpy as np
    from k3d.colormaps import paraview_color_maps

    plot = k3d.plot(grid_visible=False,
                    camera_auto_fit=False)

    bodies_count = 40
    bodies = np.random.random_sample((bodies_count, 7)).astype(np.float32)

    bodies[:, 0:6] -= 0.5
    bodies[:, 3:6] *= 0.05
    bodies[:, 6] = (bodies[:, 6] + 0.5) * 1000
    bodies[0, :] = np.array([0, 0, 0, 0, 0, 0, 1e6])

    for i in range(1, bodies_count):
        bodies[i, 0:3] = (bodies[i, 0:3] / np.linalg.norm(bodies[i, 0:3])) * 0.5

    points = k3d.points(bodies[:, 0:3],
                        point_size=0.03,
                        color=0x3e3a3a)
    plot += points

    G = 6.67E-11
    lines = []
    speeds = []
    positions = {}

    for i in range(bodies_count):
        lines.append([])
        speeds.append([])

    for t in range(500):
        for i in range(bodies_count):
            sum_force = np.zeros(3)

            for j in range(bodies_count):
                if i == j:
                    continue

                direction = bodies[j, 0:3] - bodies[i, 0:3]
                force = G * bodies[i, 6] * bodies[j, 6] * direction
                force = force / (np.linalg.norm(direction) ** 3)
                sum_force = sum_force + force

            bodies[i, 3:6] = bodies[i, 3:6] + sum_force / bodies[i, 6]

        for i in range(bodies_count):
            bodies[i, 0:3] = bodies[i, 0:3] + bodies[i, 3:6] * 0.15
            lines[i].append(np.copy(bodies[i, 0:3]))
            speeds[i].append(np.linalg.norm(bodies[i, 3:6]))

        positions[str(t * 0.01)] = np.copy(bodies[:, 0:3]).astype(np.float32)

    for line, speed in zip(lines, speeds):
        plot += k3d.line(np.array(line).astype(np.float32),
                         width=0.0002,
                         attribute=speed,
                         color_range=[0, 0.1],
                         color_map=paraview_color_maps.Erdc_iceFire_H)

    points.positions = positions

    plot.display()

    plot.camera= [1.5491, -1.2661, -0.3120,
                  -0.1189, 0.0576, -0.1350,
                  0.6329, 0.7390, -0.2306]

    plot.start_auto_play()

.. k3d_plot ::
  :filename: plots/orbits_plot.py