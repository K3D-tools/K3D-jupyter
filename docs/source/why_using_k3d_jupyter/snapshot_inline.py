import k3d
import numpy as np


def generate():
    np.random.seed(0)
    points_number = 500
    positions = 50 * np.random.random_sample((points_number, 3)) - 25
    colors = np.random.randint(0, 0xFFFFFF, points_number)

    plot = k3d.plot()
    points = k3d.points(positions.astype(np.float32), colors.astype(np.uint32), point_size=3.0,
                        shader='3dSpecular')
    plot += points

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
