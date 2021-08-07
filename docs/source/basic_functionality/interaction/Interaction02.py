import numpy as np
import k3d


def generate():
    np.random.seed(0)

    x = np.random.randn(100, 3).astype(np.float32)
    plot = k3d.plot(name='points')
    plt_points = k3d.points(positions=x, point_size=0.2, shader='3d')
    plt_points.positions = {str(t): x - t / 10 * x / np.linalg.norm(x, axis=-1)[:, np.newaxis] for t
                            in range(10)}

    plot += plt_points

    plot.snapshot_type = 'inline'

    return plot.get_snapshot(additional_js_code='K3DInstance.startAutoPlay()')
