import numpy as np
import k3d
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    plot = k3d.plot(screenshot_scale=1.0)
    headless = k3d_remote(plot, get_headless_driver(), width=320, height=226)

    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100, dtype=np.float32)
    z = np.linspace(-2, 2, 100, dtype=np.float32)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    line = k3d.line(np.vstack([x, y, z]).T, width=0.2, scaling=[1, 1, 2])

    plot += line

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1.0)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
