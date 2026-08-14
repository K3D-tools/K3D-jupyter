import unittest
import zlib

import msgpack
import numpy as np

from ..factory import mesh, plot, points, voxel_chunk

CAMERA = [3.0, -4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class TestBinarySnapshot(unittest.TestCase):
    """get_binary_snapshot / load_binary_snapshot round-trips.

    Both directions have to stay symmetric: everything the snapshot stores - the objects, the
    plot parameters and the chunkList - has to come back on load.
    """

    @staticmethod
    def _scene():
        p = plot(background_color=0xEEDDCC, grid_visible=False, camera_fov=55.0)
        p.camera = list(CAMERA)
        p += points(
            np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32), point_size=0.3
        )
        p += mesh(
            np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
            np.array([0, 1, 2], dtype=np.uint32),
        )
        return p

    def test_custom_environment_round_trips(self):
        source = self._scene()
        env = np.random.default_rng(7).random((8, 16, 3), dtype=np.float32)
        source.environment = env
        source.renderer = "advanced"

        restored = plot()
        restored.load_binary_snapshot(source.get_binary_snapshot(1))

        assert restored.renderer == "advanced"
        np.testing.assert_array_equal(np.asarray(restored.environment), env)

    def test_carries_format_version(self):
        from .._version import __version__

        data = msgpack.unpackb(zlib.decompress(self._scene().get_binary_snapshot(1)))

        assert data["version"] == __version__

    def test_restores_plot_params(self):
        source = self._scene()
        data = source.get_binary_snapshot()
        saved = source.get_plot_params()

        restored = plot()  # a fresh plot, whose defaults differ from the saved ones
        restored.load_binary_snapshot(data)

        self.assertEqual(len(restored.objects), len(source.objects))
        for key, value in saved.items():
            self.assertEqual(restored.get_plot_params()[key], value, msg=key)

    def test_accepts_numpy_plot_params(self):
        p = plot()
        p.grid = np.array([-1, -1, -1, 1, 1, 1])
        p.camera = np.array(CAMERA, dtype=np.float32)

        self.assertGreater(len(p.get_binary_snapshot()), 0)

    def test_preserves_voxel_chunks(self):
        chunk = voxel_chunk(np.ones((2, 2, 2), dtype=np.uint8), coord=[0, 0, 0])
        source = plot()
        data = source.get_binary_snapshot(voxel_chunks=[chunk])

        restored = plot()
        restored.load_binary_snapshot(data)
        self.assertEqual(len(restored.voxel_chunks), 1)

        # Re-saving must keep the chunks without the caller passing them back in.
        resaved = msgpack.unpackb(zlib.decompress(restored.get_binary_snapshot()))
        self.assertEqual(len(resaved["chunkList"]), 1)

    def test_voxel_chunks_available_before_any_load(self):
        self.assertEqual(plot().voxel_chunks, [])
