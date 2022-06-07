import k3d


def generate():
    vertices = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    indices = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]]

    plt_tetra = k3d.mesh(vertices, indices,
                         colors=[0x32ff31, 0x37d3ff, 0xbc53ff, 0xffc700])

    plot = k3d.plot()
    plot += plt_tetra

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
