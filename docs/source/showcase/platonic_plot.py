import k3d.platonic as platonic
import math
import k3d


def generate():
    plot = k3d.plot(screenshot_scale=1.0)

    meshes = [
        platonic.Dodecahedron().mesh,
        platonic.Cube().mesh,
        platonic.Icosahedron().mesh,
        platonic.Octahedron().mesh,
        platonic.Tetrahedron().mesh
    ]

    colors = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff]

    for i, obj in enumerate(meshes):
        rad = math.radians(i / len(meshes) * 360)
        radius = 3.5
        obj.transform.translation = [math.sin(rad) * radius, math.cos(rad) * radius, 0]
        obj.color = colors[i]
        plot += obj

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
