import k3d
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    plt_square = k3d.mesh(vertices=[[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]],
                          indices=[[0, 1, 2], [2, 1, 0], [0, 2, 3], [3, 2, 0]],
                          colors=[0xff0000, 0x00ff00, 0x0000ff, 0xffff00])

    plt_text_lt = k3d.text('Left-Top',
                           position=(-1, 0, 1), reference_point='lt',
                           color=0xffff00, size=0.45)
    plt_text_lb = k3d.text('Left-Bottom',
                           position=(-1, 0, -1), reference_point='lb',
                           color=0x0000ff, size=0.45)
    plt_text_lc = k3d.text('Left-Center',
                           position=(-1, 0, 0), reference_point='lc',
                           color=0x808080, size=0.45)
    plt_text_rt = k3d.text('Right-Top',
                           position=(1, 0, 1), reference_point='rt',
                           color=0xff0000, size=0.45)
    plt_text_rb = k3d.text('Right-Bottom',
                           position=(1, 0, -1), reference_point='rb',
                           color=0x00ff00, size=0.45)
    plt_text_rc = k3d.text('Right-Center',
                           position=(1, 0, 0), reference_point='rc',
                           color=0x808000, size=0.45)
    plt_text_ct = k3d.text('Center-Top',
                           position=(0, 0, 1), reference_point='ct',
                           color=0xff8000, size=0.45)
    plt_text_cb = k3d.text('Center-Bottom',
                           position=(0, 0, -1), reference_point='cb',
                           color=0x008080, size=0.45)
    plt_text_cc = k3d.text('Center-Center',
                           position=(0, 0, 0), reference_point='cc',
                           color=0xff00ff, size=0.45)

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)
    plot += plt_square
    plot += plt_text_lt
    plot += plt_text_lb
    plot += plt_text_lc
    plot += plt_text_rt
    plot += plt_text_rb
    plot += plt_text_rc
    plot += plt_text_ct
    plot += plt_text_cb
    plot += plt_text_cc

    plot.camera = [0, -2.5, 0,
                   0, 0, 0,
                   0, 0, 1]

    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
