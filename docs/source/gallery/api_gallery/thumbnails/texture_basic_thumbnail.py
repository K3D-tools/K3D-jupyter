import k3d
import os
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../../api_reference/plot_objects/assets/arcade_carpet_512.png')

    with open(filepath, 'rb') as stl:
        data = stl.read()

    plt_texture = k3d.texture(data,
                              file_format='png')

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)
    plot += plt_texture

    plot.camera = [0, 0, 1.5,
                   0, 0, 0,
                   0, 1, 0]

    headless = k3d_remote(plot, get_headless_driver(), width=600, height=370)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
