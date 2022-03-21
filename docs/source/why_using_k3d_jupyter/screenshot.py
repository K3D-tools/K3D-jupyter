import k3d
import numpy as np
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    plot = k3d.plot()
    headless = k3d_remote(plot, get_headless_driver())

    np.random.seed(0)
    points_number = 500

    positions = 50 * np.random.random_sample((points_number, 3)) - 25
    colors = np.random.randint(0, 0xFFFFFF, points_number)

    points = k3d.points(
        positions.astype(np.float32), colors.astype(np.uint32),
        point_size=3.0,
        shader='3dSpecular'
    )
    plot += points

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1.0)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
