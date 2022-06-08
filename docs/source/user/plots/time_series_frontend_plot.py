import numpy as np

import k3d


def generate():
    np.random.seed(2022)
    x = np.random.randn(100,3).astype(np.float32)

    plt_points = k3d.points(x,
                            color=0x528881,
                            point_size=0.2)

    plot = k3d.plot()
    plot += plt_points

    plt_points.positions = {str(t):x - t/5*x/np.linalg.norm(x,axis=-1)[:,np.newaxis] for t in range(10)}

    plot.snapshot_type = 'inline'
    return plot.get_snapshot(additional_js_code='K3DInstance.startAutoPlay()')
