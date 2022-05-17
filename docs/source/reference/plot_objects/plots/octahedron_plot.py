import k3d
from k3d import platonic


def generate():
    plot = k3d.plot()

    octa_1 = platonic.Octahedron()
    octa_2 = platonic.Octahedron(origin=[5, -2, 3], size=0.5)

    plot += octa_1.mesh
    plot += octa_2.mesh

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
