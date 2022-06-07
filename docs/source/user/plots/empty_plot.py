import k3d


def generate():
    plot = k3d.plot()

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
