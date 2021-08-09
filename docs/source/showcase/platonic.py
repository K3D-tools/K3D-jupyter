import k3d.platonic as platonic
import math
import k3d
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    plot = k3d.plot(screenshot_scale=1.0)
    headless = k3d_remote(plot, get_headless_driver(), width=320, height=226)

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

    plot.camera = [1.0, 1.0, 1.5,
                   0, 0, 0,
                   0, 0, 1]
    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1.0)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
