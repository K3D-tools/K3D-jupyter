import k3d
import os


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../assets/factory/skull_w_jaw.stl')

    with open(filepath, 'rb') as stl:
        data = stl.read()

    plt_skull = k3d.stl(data, color=0xe3dac9)

    plot = k3d.plot()
    plot += plt_skull

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
