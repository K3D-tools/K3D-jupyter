import k3d
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    plot = k3d.plot(screenshot_scale=1.0)
    headless = k3d_remote(plot, get_headless_driver())

    plot += k3d.mesh([0, 1, 1,
                      1, 1, 0,
                      1, 1, 1,

                      1, 2, 2,
                      1, 1, 1,
                      2, 1, 1], [0, 1, 2, 3, 4, 5], color=0x00ff00)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1.0)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
