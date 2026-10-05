import io

import numpy as np
from PIL import Image

import k3d


def png(pixels):
    buffer = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(buffer, format='PNG')
    return buffer.getvalue()


def generate():
    # a tile in the x-z plane, facing the default camera
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)

    # cells of 0..1 across a 128 px image, four by four
    y, x = (np.mgrid[0:128, 0:128] + 0.5) / 32 % 1.0 - 0.5
    r = np.hypot(x, y)

    # bumps: a tangent-space normal map, green pointing up the image
    inside = r < 0.4
    nx, ny = np.where(inside, x / 0.4, 0), np.where(inside, -y / 0.4, 0)
    nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))
    bumps = (np.stack([nx, ny, nz], -1) * 0.5 + 0.5) * 255

    # glowing rings, and discs cut out of a plate
    rings = np.where(np.abs(r - 0.3) < 0.06, 255, 0)[..., None] * np.array([1.0, 0.6, 0.2])
    discs = np.dstack([np.full(r.shape, 60), np.full(r.shape, 140), np.full(r.shape, 230),
                       np.where(r < 0.35, 0, 255)])

    plot = k3d.plot(renderer='advanced', grid_visible=False)

    plot += k3d.mesh(vertices, indices, uvs=uvs, color=0xB0B8C0, metalness=0.6,
                     roughness=0.3, normal_map=png(bumps), flat_shading=False,
                     name='normal map')
    plot += k3d.mesh(vertices + [1.1, 0, 0], indices, uvs=uvs, color=0x202020,
                     emissive=0xFFFFFF, emissive_map=png(rings), emissive_intensity=1.5,
                     name='emissive')
    plot += k3d.mesh(vertices + [2.2, 0, 0], indices, uvs=uvs, texture=png(discs),
                     alpha_mode='mask', side='double', name='alpha mask')

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
