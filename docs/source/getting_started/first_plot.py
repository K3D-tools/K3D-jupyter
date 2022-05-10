import k3d

def generate():
    plot = k3d.plot()
    plot += k3d.line([[0, 0, 0],
                      [1, 1, 1]])

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
