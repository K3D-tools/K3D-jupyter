"""The mesh material of 3.2.0, rendered: colour composition, emissive, alpha modes, the extra maps.

Every texture here is drawn by the test itself, so what each pixel should be is known from the
code below and not from an image file. The surfaces stand in the x-z plane, facing the default
camera, which looks from -y.
"""

import io

import numpy as np
import pytest
from PIL import Image

import k3d

from .plot_compare import compare, prepare

# a unit quad in the x-z plane, normal towards -y, uv (0, 0) at the bottom left
QUAD_VERTICES = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], np.float32)
QUAD_INDICES = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
# glTF's convention, which K3D textures follow: v = 0 is the first row of the image, the top
QUAD_UVS = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)


def png(pixels):
    """Encoded PNG of an (h, w, 3|4) uint8 array."""
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(pixels, np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def checker(size=64, cells=8, a=(255, 255, 255), b=(40, 40, 40)):
    y, x = np.mgrid[0:size, 0:size] * cells // size
    mask = ((x + y) % 2 == 0)[..., None]
    return np.where(mask, np.array(a, np.uint8), np.array(b, np.uint8))


def quad(offset=(0, 0, 0), scale=1.0, uvs=QUAD_UVS, **kwargs):
    return k3d.mesh(QUAD_VERTICES * scale + np.array(offset, np.float32), QUAD_INDICES,
                    uvs=uvs, side="double", **kwargs)


def disc_alpha(size=64):
    """RGBA: an orange disc, opaque in the middle and fading to nothing at its rim."""
    y, x = (np.mgrid[0:size, 0:size] + 0.5) / size - 0.5
    alpha = np.clip(1.0 - np.hypot(x, y) / 0.5, 0, 1)
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 255, 140, 0
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    return rgba


def bumps(size=128, count=3):
    """Tangent-space normal map of a grid of hemispherical bumps."""
    y, x = (np.mgrid[0:size, 0:size] + 0.5) / size * count % 1.0 - 0.5
    r = np.hypot(x, y)
    inside = r < 0.4
    nx = np.where(inside, x / 0.4, 0.0)
    # image rows run down while v runs up the bump: green points up the image
    ny = np.where(inside, -y / 0.4, 0.0)
    nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))
    return ((np.stack([nx, ny, nz], -1) * 0.5 + 0.5) * 255).astype(np.uint8)


def backdrop():
    """A grey wall behind the samples, so transparency has something to show."""
    return quad(offset=(-0.25, 0.6, -0.25), scale=3.5, color=0x8899AA, uvs=None)


def test_mesh_colour_composition():
    prepare()

    gradient = np.array([0xFFFFFF, 0xFF0000, 0x00FF00, 0x0000FF], np.uint32)

    # texture alone, texture x vertex colours, texture x vertex colours x base colour
    pytest.plot += quad((0, 0, 0), texture=png(checker()))
    pytest.plot += quad((1.1, 0, 0), texture=png(checker()), colors=gradient)
    pytest.plot += quad((2.2, 0, 0), texture=png(checker()), colors=gradient, color=0x80FFFF)
    # a colormap next to a texture multiplies it too
    pytest.plot += quad((3.3, 0, 0), texture=png(checker()), attribute=[0, 1, 1, 0],
                        color_map=k3d.basic_color_maps.CoolWarm, color_range=[0, 1])

    compare("mesh_colour_composition")


def test_mesh_emissive():
    prepare()

    # a box open towards the camera; the glowing panel sits deep in it, where occlusion is darkest
    pytest.plot += k3d.mesh(
        np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
                  [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2]], np.float32),
        np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 3, 7], [0, 7, 4],
                  [1, 5, 6], [1, 6, 2], [3, 2, 6], [3, 6, 7]], np.uint32),
        color=0x707070, side="double", flat_shading=True,
    )
    pattern = np.zeros((32, 32, 3), np.uint8)
    pattern[4:28, 4:28] = (255, 220, 120)
    pytest.plot += quad((0.5, 1.95, 0.5), color=0x101010, emissive=0xFFFFFF, emissive_intensity=1.0,
                        emissive_map=png(pattern))
    pytest.plot.camera = [1, -4, 1, 1, 1, 1, 0, 0, 1]

    compare("mesh_emissive", camera_factor=None)


def test_mesh_alpha_modes():
    prepare()

    disc = png(disc_alpha())
    pytest.plot += backdrop()
    # opaque ignores the alpha, blend fades with it, mask cuts at half of it
    pytest.plot += quad((0, 0, 0), texture=disc, alpha_mode="opaque")
    pytest.plot += quad((1.1, 0, 0), texture=disc, alpha_mode="blend")
    pytest.plot += quad((2.2, 0, 0), texture=disc, alpha_mode="mask", alpha_cutoff=0.5)
    # alpha per vertex: a fade from the left edge to the right one
    pytest.plot += quad((0, 0, -1.1), color=0x2060FF, colors=np.array(
        [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]], np.float32))
    pytest.plot += quad((1.1, 0, -1.1), color=0x2060FF, opacities=[0, 1, 1, 0], alpha_mode="mask",
                        alpha_cutoff=0.5)

    compare("mesh_alpha_modes")


def test_mesh_normal_map():
    prepare()

    flat = np.full((4, 4, 3), (128, 128, 255), np.uint8)
    pytest.plot += quad((0, 0, 0), color=0xC0C0C0, normal_map=png(bumps()), flat_shading=False)
    pytest.plot += quad((1.1, 0, 0), color=0xC0C0C0, normal_map=png(bumps()), normal_scale=-1.0,
                        flat_shading=False)
    pytest.plot += quad((2.2, 0, 0), color=0xC0C0C0, normal_map=png(flat), flat_shading=False)

    compare("mesh_normal_map")


def test_mesh_metalness_roughness_and_occlusion_maps():
    prepare()

    # green = roughness, blue = metalness, in four stripes: (rough, metal) 0/0, 1/0, 0/1, 1/1
    stripes = np.zeros((4, 64, 3), np.uint8)
    for i, (g, b) in enumerate([(20, 0), (255, 0), (20, 255), (255, 255)]):
        stripes[:, i * 16:(i + 1) * 16, 1] = g
        stripes[:, i * 16:(i + 1) * 16, 2] = b
    pytest.plot += quad((0, 0, 0), color=0xD0A060, metalness=1.0, roughness=1.0,
                        metalness_roughness_map=png(stripes), flat_shading=False)

    # occlusion: a dark ring read with a second uv set that shows it at half size
    y, x = (np.mgrid[0:64, 0:64] + 0.5) / 64 - 0.5
    ring = np.where(np.abs(np.hypot(x, y) - 0.3) < 0.08, 0, 255).astype(np.uint8)
    occlusion = np.stack([ring, ring, ring], -1)
    pytest.plot += quad((1.1, 0, 0), color=0xFFFFFF, occlusion_map=png(occlusion),
                        uvs2=QUAD_UVS * 2.0 - 0.5, texture_wrap="clamp")
    pytest.plot += quad((2.2, 0, 0), color=0xFFFFFF, occlusion_map=png(occlusion),
                        occlusion_strength=0.5)

    compare("mesh_metalness_roughness_occlusion")


def test_mesh_texture_wrap_and_mipmaps():
    prepare()

    arrow = np.zeros((32, 32, 3), np.uint8)
    arrow[...] = (230, 230, 230)
    arrow[:, :6] = (200, 30, 30)
    arrow[:6, :] = (30, 30, 200)
    arrow[12:20, 8:24] = (20, 140, 20)
    tiled = QUAD_UVS * 3.0 - 1.0
    for i, wrap in enumerate(["clamp", "repeat", "mirror", "repeat clamp"]):
        pytest.plot += quad((i * 1.1, 0, 0), texture=png(arrow), uvs=tiled, texture_wrap=wrap)

    # a checkerboard far finer than the pixels it lands on: mipmapped, it averages to grey
    pytest.plot += k3d.mesh(
        np.array([[0, -0.2, -0.2], [4.3, -0.2, -0.2], [4.3, 6, -0.2], [0, 6, -0.2]], np.float32),
        QUAD_INDICES, uvs=QUAD_UVS * 40, texture=png(checker(64, 2)), texture_wrap="repeat",
        side="double",
    )

    compare("mesh_texture_wrap_mipmaps")


def test_points_colour_composition():
    prepare()

    positions = np.array([[x, 0, z] for z in range(2) for x in range(4)], np.float32)
    colors = np.array([0xFFFFFF, 0xFF0000, 0x00FF00, 0x0000FF] * 2, np.uint32)

    for i, shader in enumerate(["3d", "flat", "mesh"]):
        offset = np.array([0, 0, i * 2.5], np.float32)
        # colours alone, colours x colour, colormap x colour
        pytest.plot += k3d.points(positions[:4] + offset, colors=colors[:4], shader=shader,
                                  point_size=0.6)
        pytest.plot += k3d.points(positions[:4] + offset + [0, 0, 0.8], colors=colors[:4],
                                  color=0x808080, shader=shader, point_size=0.6)
        pytest.plot += k3d.points(positions[:4] + offset + [0, 0, 1.6], attribute=[0, 1, 2, 3],
                                  color_map=k3d.basic_color_maps.Jet, color_range=[0, 3],
                                  color=0x80FF80, shader=shader, point_size=0.6)

    compare("points_colour_composition")


def test_vectors_colour_composition():
    prepare()

    origins = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32)
    vectors = np.array([[0, 0, 1]] * 3, np.float32)
    colors = np.array([0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFFFFFF, 0xFF00FF], np.uint32)

    pytest.plot += k3d.vectors(origins, vectors, colors=colors, line_width=0.05)
    pytest.plot += k3d.vectors(origins + [0, 0, 1.5], vectors, colors=colors, origin_color=0x808080,
                               head_color=0x00FFFF, line_width=0.05)

    compare("vectors_colour_composition")


def sphere(center, radius=0.45, n=48):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2))
    unit = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], -1).reshape(-1, 3)
    faces = [[i * n + j, (i + 1) * n + j, i * n + j + 1] for i in range(n // 2 - 1) for j in range(n - 1)]
    faces += [[i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1] for i in range(n // 2 - 1) for j in range(n - 1)]
    return (unit * radius + center).astype(np.float32), np.array(faces, np.uint32), unit.astype(np.float32)


def test_mesh_transmission():
    prepare()

    pytest.plot += quad((-0.3, 0.8, -0.3), scale=3.0, texture=png(checker(64, 6)))
    # clear glass, a denser gem, and tinted glass that darkens with thickness
    for i, (ior, color, distance) in enumerate([(1.5, 0xFFFFFF, 0.0), (2.4, 0xFFFFFF, 0.0),
                                                (1.5, 0x60A0FF, 0.5)]):
        vertices, indices, normals = sphere([0.3 + i * 0.9, 0, 0.9])
        pytest.plot += k3d.mesh(vertices, indices, normals=normals, flat_shading=False, color=0xFFFFFF,
                                roughness=0.05, transmission=1.0, ior=ior, thickness=0.9,
                                attenuation_color=color, attenuation_distance=distance)

    compare("mesh_transmission")
