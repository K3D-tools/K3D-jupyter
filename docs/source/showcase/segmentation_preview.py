import numpy as np
import k3d
from k3d.headless import k3d_remote, get_headless_driver
import pathlib

path = pathlib.Path(__file__).parent.resolve()


def generate():
    plot = k3d.plot(screenshot_scale=1.0, camera_auto_fit=False)
    headless = k3d_remote(plot, get_headless_driver(), width=320, height=226)

    data = np.load(str(path) + '/assets/segmentations.npz')

    voxels_classic = (data['seg'] + data['GT'] * 2).astype(np.uint8)
    bbox = [
        0, data['seg'].shape[2] * data['spacing'][0],
        0, data['seg'].shape[1] * data['spacing'][1],
        0, data['seg'].shape[0] * data['spacing'][2],
    ]

    plot += k3d.voxels(voxels_classic, bounds=bbox, color_map=(0xff0000, 0x00ff00, 0xffff00))

    plot += k3d.text2d('True Positive',
                       position = [0.05, 0.0],
                       reference_point='lt',
                       size=1.0,
                       color=0xaaaa00)

    plot += k3d.text2d('False Positive',
                       position = [0.05, 0.20],
                       reference_point='lt',
                       size=1.0,
                       color=0xff0000)

    plot += k3d.text2d('False Negative',
                       position = [0.05, 0.40],
                       reference_point='lt',
                       size=1.0,
                       color=0x00ff00)

    plot.camera = [
        149.62, 71.26, 146.03,
        112.18, 111.05, 73.53,
        -0.52, 0.55, 0.64
    ]

    headless.sync(hold_until_refreshed=True)


    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
