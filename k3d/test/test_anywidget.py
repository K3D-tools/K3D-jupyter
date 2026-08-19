import numpy as np

import k3d
from k3d.objects.base import VoxelChunk

VERTICES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
INDICES = np.array([[0, 1, 2]], dtype=np.uint32)


def test_esm_is_packaged():
    from k3d._widget import _STATIC

    assert (_STATIC / "widget.mjs").exists()


def test_no_widget_carries_the_module_in_its_state():
    # _esm rides in the synced state of every instance, so nothing may carry the module
    # itself: objects get a stub, the plot and the editor a loader that asks the kernel
    assert len(str(k3d.mesh(VERTICES, INDICES)._esm)) < 4096
    assert len(str(k3d.plot()._esm)) < 4096
    assert len(str(k3d.transfer_function_editor()._esm)) < 4096


def test_the_module_is_served_over_the_comm():
    from k3d._widget import _MODULE

    plot = k3d.plot()
    sent = []
    plot.send = lambda content, buffers=None: sent.append((content, buffers))

    plot._handle_custom_msg({"msg_type": "fetch_widget_module"}, [])

    (content, buffers), = sent
    assert content["msg_type"] == "widget_module"
    assert buffers[0] == _MODULE.read_bytes()


def test_widget_kinds():
    assert k3d.plot()._kind == "plot"
    assert k3d.mesh(VERTICES, INDICES)._kind == "object"
    assert VoxelChunk(voxels=np.ones((2, 2, 2), dtype=np.uint8),
                      coord=np.zeros(3, dtype=np.uint32), multiple=1)._kind == "chunk"
    assert k3d.transfer_function_editor()._kind == "tf_editor"


def test_synced_props_enumerate_public_traits():
    mesh = k3d.mesh(VERTICES, INDICES)

    assert "vertices" in mesh._synced_props
    assert "indices" in mesh._synced_props
    assert "compression_level" in mesh._synced_props
    assert not any(name.startswith("_") for name in mesh._synced_props)


def test_binary_snapshot_carries_no_transport_keys():
    mesh = k3d.mesh(VERTICES, INDICES)

    assert not any(key.startswith("_") for key in mesh.get_binary())


def test_snapshot_roundtrip_strips_transport_keys():
    plot = k3d.plot()
    plot += k3d.mesh(VERTICES, INDICES)

    snapshot = plot.get_binary_snapshot(1)

    plot2 = k3d.plot()
    plot2.load_binary_snapshot(snapshot)

    assert len(plot2.objects) == 1
    assert np.allclose(plot2.objects[0].vertices, VERTICES)


def test_synced_props_ride_in_the_initial_state():
    # ipywidgets opens the comm inside __init__ - the list must resolve on the
    # first get_state, or the front-end stub adopts a model with no attributes
    state = k3d.mesh(VERTICES, INDICES).get_state()

    assert "vertices" in state["_synced_props"]
    assert "id" in state["_synced_props"]


def test_snapshot_source_served_over_comm():
    import zlib

    plot = k3d.plot()
    sent = []
    plot.send = lambda content, buffers=None: sent.append((content, buffers))

    plot._handle_custom_msg({"msg_type": "fetch_snapshot_source"}, [])

    assert sent[0][0]["msg_type"] == "snapshot_source"
    source = zlib.decompress(sent[0][1][0])
    assert b"CreateK3DAndLoadBinarySnapshot" in source


def _relay_plot_with_mesh():
    plot = k3d.plot()
    mesh = k3d.mesh(VERTICES, INDICES)
    plot += mesh

    sent = []
    plot.send = lambda content, buffers=None: sent.append((content, buffers))

    return plot, mesh, sent


def test_relay_serves_objects_state_over_the_plot_comm():
    import zlib

    import msgpack

    plot, mesh, sent = _relay_plot_with_mesh()
    plot._relay_send_state([mesh.id])

    assert sent[-1][0] == {"msg_type": "objects_state"}
    state = msgpack.unpackb(zlib.decompress(sent[-1][1][0]), strict_map_key=False)
    assert [o["id"] for o in state["objects"]] == [mesh.id]
    assert state["objects"][0]["type"] == "Mesh"
    assert "vertices" in state["objects"][0]


def test_relay_forwards_trait_changes_as_patches():
    import zlib

    import msgpack

    plot, mesh, sent = _relay_plot_with_mesh()
    plot._relay_send_state([mesh.id])
    sent.clear()

    mesh.visible = False

    patches = [s for s in sent if s[0] == {"msg_type": "object_patch"}]
    assert len(patches) == 1
    patch = msgpack.unpackb(zlib.decompress(patches[0][1][0]), strict_map_key=False)
    assert patch["id"] == mesh.id
    assert patch["key"] == "visible"
    assert patch["value"] is False


def test_relay_applies_frontend_object_change():
    import zlib

    import msgpack

    plot, mesh, sent = _relay_plot_with_mesh()
    payload = msgpack.packb(
        {"id": mesh.id, "key": "visible", "value": False}, use_bin_type=True
    )

    plot._relay_apply_change([zlib.compress(payload, 1)])

    assert mesh.visible is False


def test_relay_applies_binary_values_via_from_json():
    import zlib

    import msgpack

    plot, mesh, sent = _relay_plot_with_mesh()
    moved = (VERTICES + 1.0).astype(np.float32)
    payload = msgpack.packb(
        {
            "id": mesh.id,
            "key": "vertices",
            "value": {
                "data": moved.tobytes(),
                "dtype": "float32",
                "shape": list(moved.shape),
            },
        },
        use_bin_type=True,
    )

    plot._relay_apply_change([zlib.compress(payload, 1)])

    np.testing.assert_array_equal(mesh.vertices, moved)


def test_ao_params_are_validated():
    import pytest as pt
    from traitlets import TraitError

    plot = k3d.plot()
    assert plot.ao_radius == 0.07 and plot.ao_strength == 1.8

    plot.ao_radius = 0.3
    plot.ao_strength = 0.0

    with pt.raises(TraitError):
        plot.ao_radius = 0.0
    with pt.raises(TraitError):
        plot.ao_radius = 1.5
    with pt.raises(TraitError):
        plot.ao_strength = -1.0
    with pt.raises(TraitError):
        plot.ao_strength = 11.0


def test_cinematic_params_are_validated():
    import pytest as pt
    from traitlets import TraitError

    plot = k3d.plot()
    assert plot.renderer == "simple"
    assert plot.cinematic_samples == 64 and plot.cinematic_bounces == 6

    plot.renderer = "cinematic"
    plot.cinematic_samples = 32
    plot.cinematic_bounces = 12

    plot.cinematic_samples = 4096

    with pt.raises(TraitError):
        plot.cinematic_samples = 0
    with pt.raises(TraitError):
        plot.cinematic_samples = 100001
    with pt.raises(TraitError):
        plot.cinematic_bounces = 0
    with pt.raises(TraitError):
        plot.cinematic_bounces = 33

    # on by default: metal under a bright sun throws fireflies that no budget clears
    assert plot.cinematic_glossy_filter == 0.25
    plot.cinematic_glossy_filter = 0.0

    with pt.raises(TraitError):
        plot.cinematic_glossy_filter = -0.1
    with pt.raises(TraitError):
        plot.cinematic_glossy_filter = 1.1

    # a plot parameter is dead in the headless and snapshot paths until it is listed in
    # _PLOT_PARAMS, which no renderer test can catch
    assert plot.get_plot_params()["cinematicGlossyFilter"] == 0.0


def test_cinematic_params_reach_the_snapshot():
    plot = k3d.plot(renderer="cinematic", cinematic_samples=16, cinematic_bounces=4)

    params = plot.get_plot_params()

    assert params["renderer"] == "cinematic"
    assert params["cinematicSamples"] == 16
    assert params["cinematicBounces"] == 4
