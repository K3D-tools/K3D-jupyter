import k3d
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    plt_texture_text = k3d.texture_text('Texture',
                                        position=[0, 0, 0],
                                        font_face='Calibri',
                                        font_weight=600,
                                        color=0xa2ffc8)

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)
    plot += plt_texture_text

    headless = k3d_remote(plot, get_headless_driver(), width=600, height=370)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(0.5)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot

