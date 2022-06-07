import k3d
from k3d import platonic


def generate():
    plot = k3d.plot()

    cube_1 = platonic.Cube()
    cube_2 = platonic.Cube(origin=[5, -2, 3], size=0.5)

    plot += cube_1.mesh
    plot += cube_2.mesh

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
