import numpy as np
import k3d
import pathlib

path = pathlib.Path(__file__).parent.resolve()


def generate():
    plot = k3d.plot(screenshot_scale=1.0, camera_auto_fit=False)

    data = np.load(str(path) + '/assets/points_cloud.npz')['arr_0']
    plot += k3d.points(data[:, 0:3],
                       data[:, 4].astype(np.uint32), point_size=0.15, shader="flat")
    plot.camera = [20.84, -3.06, 6.96,
                   0.67, 0.84, 3.79,
                   0.0, 0.0, 1.0]

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
