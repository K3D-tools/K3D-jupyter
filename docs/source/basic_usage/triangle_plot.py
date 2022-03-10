import k3d


def generate():
    plot = k3d.plot()

    vertices = [[0, 0, 0], [1, 0, 0], [0, 0, 1]]
    indices = [[0, 1, 2]]

    mesh = k3d.mesh(vertices, indices)

    plot += mesh

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
