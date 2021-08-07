import numpy as np
import k3d


def generate():
    plot = k3d.plot(name='Wiener process')
    N = 1000
    traj = np.cumsum(np.random.randn(N, 3).astype(np.float32), axis=0)
    plt_line = k3d.line(traj, shader='mesh', width=0.5)
    plt_line2 = k3d.line([traj[0], traj[-1]], shader='mesh', width=.5, color=0xff0000)
    plot += plt_line
    plot += plt_line2

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
