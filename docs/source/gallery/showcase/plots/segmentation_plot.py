import numpy as np
import os

import k3d


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../assets/segmentations.npz')

    data = np.load(filepath)

    voxels_classic = (data['seg'] + data['GT'] * 2).astype(np.uint8)
    bbox = [0, data['seg'].shape[2] * data['spacing'][0],
            0, data['seg'].shape[1] * data['spacing'][1],
            0, data['seg'].shape[0] * data['spacing'][2]]

    plt_voxels = k3d.voxels(voxels_classic,
                            bounds=bbox,
                            color_map=(0xff7764, 0x3f71d8, 0x13c08d))

    plt_tp_text = k3d.text2d('True Positive',
                             position=[0.01, 0.05],
                             reference_point='lc',
                             is_html=True,
                             color=0x3f71d8)

    plt_fp_text = k3d.text2d('False Positive',
                             position=[0.01, 0.12],
                             is_html=True,
                             reference_point='lc',
                             color=0x13c08d)

    plt_fn_text = k3d.text2d('False Negative',
                             position=[0.01, 0.19],
                             is_html=True,
                             reference_point='lc',
                             color=0xff7764)

    plot = k3d.plot(grid_visible=False,
                    camera_auto_fit=False)
    plot += plt_voxels
    plot += plt_tp_text
    plot += plt_fp_text
    plot += plt_fn_text

    plot.camera = [197.8722, 74.5292, 122.9294,
                   62.2292, 85.1011, 55.6868,
                   -0.3889, 0.0614, 0.9192]

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
