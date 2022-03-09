import k3d
import k3d.platonic as platonic


def generate():
    plot = k3d.plot(screenshot_scale=1.0)
    plot += platonic.Cube().mesh

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
