import io

import numpy as np
from PIL import Image

import k3d
from k3d.helpers import download

MAPS = ('texture', 'emissive_map', 'normal_map', 'metalness_roughness_map', 'occlusion_map')


def halved(data):
    """The image at half its size - a docs page carries every texture inline."""
    image = Image.open(io.BytesIO(data))
    image = image.resize((image.width // 2, image.height // 2), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def scene(renderer):
    # Model: BoomBox, Microsoft, CC0,
    # https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BoomBox
    filename = download('https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/'
                        'main/Models/BoomBox/glTF-Binary/BoomBox.glb')

    # two centimetres wide; a hundred times that reads better next to the default grid
    boombox = k3d.glb(filename, scaling=[100, 100, 100])

    for part in boombox:
        for name in MAPS:
            if getattr(part, name):
                setattr(part, name, halved(getattr(part, name)))

    bounds = np.array([part.get_bounding_box() for part in boombox])
    low, high = bounds[:, 0::2].min(axis=0), bounds[:, 1::2].max(axis=0)
    centre = (low + high) / 2
    size = float((high - low).max())

    plot = k3d.plot(renderer=renderer,
                    environment='studio',
                    tone_mapping='aces',
                    grid_visible=False,
                    camera_auto_fit=False,
                    background_color=0x1E2126,
                    cinematic_samples=256,
                    cinematic_bounces=6)
    plot += boombox

    # what the cinematic renderer bounces light off, and what the speaker grille shadows
    span = 1.5 * size
    floor_z = float(low[2])
    plot += k3d.mesh(np.array([[centre[0] - span, centre[1] - span, floor_z],
                               [centre[0] + span, centre[1] - span, floor_z],
                               [centre[0] + span, centre[1] + span, floor_z],
                               [centre[0] - span, centre[1] + span, floor_z]], np.float32),
                     np.array([[0, 1, 2], [0, 2, 3]], np.uint32),
                     color=0x3A3F47, roughness=0.6, name='floor')

    eye = centre + np.array([0.55, -1.2, 0.45]) * size
    plot.camera = [*eye, *centre, 0, 0, 1]

    return plot


def generate():
    plot = scene('advanced')

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
