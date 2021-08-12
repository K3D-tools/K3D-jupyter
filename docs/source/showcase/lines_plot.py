import numpy as np
import k3d


def generate():
    plot = k3d.plot(screenshot_scale=1.0)
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100, dtype=np.float32)
    z = np.linspace(-2, 2, 100, dtype=np.float32)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    line = k3d.line(np.vstack([x, y, z]).T, width=0.2, scaling=[1, 1, 2])

    plot += line

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
