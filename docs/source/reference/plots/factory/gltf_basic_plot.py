import os

import k3d


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../assets/factory/BoxTextured/BoxTextured.gltf')

    box = k3d.gltf(filepath)

    plot = k3d.plot()
    plot += box

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
