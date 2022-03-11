import k3d
import k3d.platonic as platonic


def generate():
    plot = k3d.plot()
    plot += platonic.Cube().mesh

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
