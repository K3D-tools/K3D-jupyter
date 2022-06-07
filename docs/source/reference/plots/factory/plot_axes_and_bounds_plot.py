import k3d


def generate():
    plot = k3d.plot(grid=(0, 0, 0, 60, 20, 50),
                    axes=['Time', 'Mass', 'Temperature'])

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
