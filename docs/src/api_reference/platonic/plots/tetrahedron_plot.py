import k3d
from k3d import platonic


def generate():
    plot = k3d.plot()

    tetra_1 = platonic.Tetrahedron()
    tetra_2 = platonic.Tetrahedron(origin=[5, -2, 3], size=0.5)

    plot += tetra_1.mesh
    plot += tetra_2.mesh

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
