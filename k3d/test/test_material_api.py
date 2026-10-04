"""The mesh material parameters of 3.2.0 and the colour rule that came with them, without a browser.

Since 3.2.0 an object's `color` multiplies its other colour sources - per-vertex colours, a
colormap, a texture - instead of being ignored next to them. A colour nobody asked for must
therefore be white wherever another source is given, or the default blue would tint it.
"""

import numpy as np
import pytest
from traitlets import TraitError

import k3d
from k3d.factory.common import _default_color
from k3d.helpers import image_format, pack_colors
from k3d.objects import Mesh

TRIANGLE = ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
PNG_HEAD = b"\x89PNG\r\n\x1a\n" + bytes(8)


@pytest.mark.parametrize("given, expected", [
    ({}, _default_color),
    ({"colors": [1, 2, 3]}, 0xFFFFFF),
    ({"attribute": [0, 1, 2]}, 0xFFFFFF),
    ({"texture": PNG_HEAD, "uvs": [[0, 0], [1, 0], [0, 1]]}, 0xFFFFFF),
    ({"colors": [1, 2, 3], "color": 0x00FF00}, 0x00FF00),
])
def test_mesh_colour_left_out(given, expected):
    assert k3d.mesh(*TRIANGLE, **given).color == expected


@pytest.mark.parametrize("factory, given, expected", [
    (lambda **kw: k3d.points([[0, 0, 0]], **kw), {}, _default_color),
    (lambda **kw: k3d.points([[0, 0, 0]], **kw), {"colors": [5]}, 0xFFFFFF),
    (lambda **kw: k3d.points([[0, 0, 0]], **kw), {"attribute": [0.5]}, 0xFFFFFF),
])
def test_points_colour_left_out(factory, given, expected):
    assert factory(**given).color == expected


def test_vectors_colours_left_out():
    plain = k3d.vectors([0, 0, 0], [1, 1, 1])
    coloured = k3d.vectors([0, 0, 0], [1, 1, 1], colors=[0xFF0000, 0x00FF00])
    tinted = k3d.vectors([0, 0, 0], [1, 1, 1], colors=[0xFF0000, 0x00FF00], head_color=0x808080)

    assert (plain.origin_color, plain.head_color) == (_default_color, _default_color)
    assert (coloured.origin_color, coloured.head_color) == (0xFFFFFF, 0xFFFFFF)
    assert (tinted.origin_color, tinted.head_color) == (0xFFFFFF, 0x808080)


def test_an_object_built_directly_gets_white_next_to_a_source():
    assert Mesh(vertices=TRIANGLE[0], indices=TRIANGLE[1], colors=[1, 2, 3]).color == 0xFFFFFF


@pytest.mark.parametrize("colors, packed, alpha", [
    (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), [0xFF0000, 0x00FF00, 0x0000FF], None),
    (np.array([[255, 128, 0, 255], [0, 0, 0, 0], [1, 2, 3, 51]], np.uint8),
     [0xFF8000, 0x000000, 0x010203], [1.0, 0.0, 0.2]),
    ([1, 2, 3], [1, 2, 3], None),
])
def test_colour_arrays_pack(colors, packed, alpha):
    result, opacities = pack_colors(colors)

    np.testing.assert_array_equal(result, packed)
    if alpha is None:
        assert opacities is None
    else:
        np.testing.assert_allclose(opacities, alpha)


def test_rgba_colours_make_a_blended_mesh():
    mesh = k3d.mesh(*TRIANGLE, colors=np.array([[1, 0, 0, 0.5]] * 3))

    np.testing.assert_allclose(mesh.opacities, [0.5] * 3)
    assert mesh.alpha_mode == "blend"
    assert k3d.mesh(*TRIANGLE, colors=np.array([[1, 0, 0, 0.5]] * 3), alpha_mode="mask").alpha_mode == "mask"


def test_opacities_cannot_come_twice():
    with pytest.raises(ValueError, match="twice"):
        k3d.mesh(*TRIANGLE, colors=np.array([[1, 0, 0, 0.5]] * 3), opacities=[1, 1, 1])


def test_opacities_one_per_vertex():
    with pytest.raises(TraitError, match="opacities"):
        k3d.mesh(*TRIANGLE, opacities=[1, 1])


@pytest.mark.parametrize("wrap", ["clamp", "repeat", "mirror", "repeat clamp", "mirror repeat"])
def test_texture_wrap_takes_one_mode_or_two(wrap):
    assert k3d.mesh(*TRIANGLE, texture_wrap=wrap).texture_wrap == wrap


@pytest.mark.parametrize("wrap", ["tile", "repeat clamp mirror", ""])
def test_texture_wrap_refuses_the_rest(wrap):
    with pytest.raises(TraitError, match="texture_wrap"):
        k3d.mesh(*TRIANGLE, texture_wrap=wrap)


def test_alpha_mode_is_one_of_three():
    with pytest.raises(TraitError, match="alpha_mode"):
        k3d.mesh(*TRIANGLE, alpha_mode="cutout")


def test_material_parameters_arrive():
    given = {
        "emissive": 0x112233, "emissive_intensity": 2.5, "emissive_map": PNG_HEAD,
        "normal_map": PNG_HEAD, "normal_scale": -0.5, "metalness_roughness_map": PNG_HEAD,
        "occlusion_map": PNG_HEAD, "occlusion_strength": 0.75, "alpha_mode": "mask",
        "alpha_cutoff": 0.3, "texture_wrap": "repeat", "transmission": 0.8, "ior": 1.33,
        "thickness": 0.5, "attenuation_color": 0x80C0FF, "attenuation_distance": 2.0,
    }
    mesh = k3d.mesh(*TRIANGLE, uvs2=[[0, 0], [1, 0], [0, 1]], **given)

    for name, value in given.items():
        assert getattr(mesh, name) == value, name
    assert mesh.uvs2.shape == (3, 2)


def test_texture_format_is_read_from_the_bytes():
    assert k3d.mesh(*TRIANGLE, texture=PNG_HEAD).texture_file_format == "png"
    assert k3d.mesh(*TRIANGLE, texture=PNG_HEAD, texture_file_format="gif").texture_file_format == "gif"


@pytest.mark.parametrize("data, expected", [
    (b"\x89PNG\r\n\x1a\n....", "png"),
    (b"\xff\xd8\xff\xe0....", "jpeg"),
    (b"GIF89a......", "gif"),
    (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "webp"),
    (b"\xabKTX 20\xbb\r\n\x1a\n", None),
    (b"", None),
    (None, None),
])
def test_image_format(data, expected):
    assert image_format(data) == expected
