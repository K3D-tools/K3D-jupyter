import k3d
from k3d import platonic


def generate():
    plot = k3d.plot()

    ico_1 = platonic.Icosahedron()
    ico_2 = platonic.Icosahedron(origin=[5, -2, 3], size=0.5)

    plot += ico_1.mesh
    plot += ico_2.mesh

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
