import os

import numpy as np

import k3d


def lerp_color(a, b, t):
    ca = np.array([(a >> 16) & 255, (a >> 8) & 255, a & 255], float)
    cb = np.array([(b >> 16) & 255, (b >> 8) & 255, b & 255], float)
    r, g, b_ = np.rint(ca + (cb - ca) * t).astype(int)
    return int((r << 16) | (g << 8) | b_)


def scene(renderer):
    # Model: "Sasha" by saber7711 on Blendswap (https://blendswap.com/blend/29574), CC-BY;
    # converted to glTF for these docs (see assets/sasha.md). Draco-compressed: needs DracoPy.
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'sasha.glb')

    ring = k3d.glb(filename, rotation=[np.pi / 4, 0, -1, 0], compression_level=9)

    # the band in rose gold, polished
    for part in ring:
        if part.custom_data.get('gltf_material') == 'Material.002':
            part.color = lerp_color(part.color, 0xC47258, 0.45)
            part.roughness = 0.05

    boxes = np.array([part.get_bounding_box() for part in ring])
    low, high = boxes[:, 0::2].min(axis=0), boxes[:, 1::2].max(axis=0)
    centre = (low + high) / 2
    size = float((high - low).max())

    plot = k3d.plot(renderer=renderer,
                    environment='brown_photostudio_02',
                    tone_mapping='aces',
                    lighting=2.0,
                    grid_visible=False,
                    camera_auto_fit=False,
                    background_color=0xE6E6E6,
                    camera_fov=30,
                    cinematic_samples=512,
                    cinematic_bounces=32)
    plot += ring

    span = 3 * size
    floor = float(low[2])
    plot += k3d.mesh(np.array([[centre[0] - span, centre[1] - span, floor],
                               [centre[0] + span, centre[1] - span, floor],
                               [centre[0] + span, centre[1] + span, floor],
                               [centre[0] - span, centre[1] + span, floor]], np.float32),
                     np.array([[0, 1, 2], [0, 2, 3]], np.uint32),
                     color=0xF2F2F2, roughness=0.5, name='floor')

    # a glowing panel overhead, out of frame: the light the stones sparkle with
    top = float(high[2]) + 1.5 * size
    plot += k3d.mesh(np.array([[centre[0] - size, centre[1] - size, top],
                               [centre[0] + size, centre[1] - size, top],
                               [centre[0] + size, centre[1] + size, top],
                               [centre[0] - size, centre[1] + size, top]], np.float32),
                     np.array([[0, 2, 1], [0, 3, 2]], np.uint32),
                     color=0x000000, emissive=0xFFFFFF, emissive_intensity=4.0, side='double',
                     name='light box')

    # looking at the stones from outside the band, a little from above
    gems = np.array([part.get_bounding_box() for part in ring if part.transmission > 0])
    target = (gems[:, 0::2].min(axis=0) + gems[:, 1::2].max(axis=0)) / 2
    out = target - centre
    out[2] = 0
    out /= np.linalg.norm(out)
    side = np.cross([0, 0, 1], out)
    eye = target + (0.9 * out - 0.75 * side + 1.0 * np.array([0, 0, 1])) * size
    plot.camera = [*eye, *target, 0, 0, 1]

    return plot


def generate():
    plot = scene('cinematic')

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
