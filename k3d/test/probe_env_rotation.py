"""Probe, not a test: does environment_rotation reach a headless cinematic render?

Both consumer paths are covered - a rotation on its own, and one alongside an object change,
which goes through a scene rebuild instead. The control renders the same rotation twice and must
come back byte-identical, or the other numbers mean nothing.

    python k3d/test/probe_env_rotation.py
"""
import hashlib
import os
import sys
from io import BytesIO

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import k3d
from k3d.headless import get_headless_driver, k3d_remote

SAMPLES = int(os.environ.get("K3D_ROT_SAMPLES", "8"))
WIDTH = int(os.environ.get("K3D_ROT_WIDTH", "640"))
HEIGHT = int(os.environ.get("K3D_ROT_HEIGHT", "360"))
VOXELS = int(os.environ.get("K3D_ROT_VOXELS", "96"))
PORT = int(os.environ.get("K3D_ROT_PORT", "8099"))


def blob(n):
    g = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    return (np.exp(-(r ** 2) / 0.35) * (0.75 + 0.25 * np.cos(18.0 * x)) * 900).astype(np.float32)


def shot(headless, png_only=True):
    headless.sync(hold_until_refreshed=True)
    png = headless.get_screenshot(True)

    return png, np.asarray(Image.open(BytesIO(png)).convert("RGB"), dtype=np.float64)


def compare(label, a, b):
    pa, ia = a
    pb, ib = b
    identical = hashlib.md5(pa).hexdigest() == hashlib.md5(pb).hexdigest()
    d = np.abs(ia - ib)
    moved = float((d.max(axis=2) > 1).mean())

    print("  %-34s %s   sredni |delta| %6.3f   pikseli zmienionych %5.1f%%   max %3.0f"
          % (label, "IDENTYCZNE" if identical else "rozne     ", d.mean(), 100 * moved, d.max()))

    return identical


def main():
    volume = k3d.volume(blob(VOXELS), alpha_coef=200, samples=128, light_scale=2.25,
                        color_range=[80, 900],
                        color_map=k3d.matplotlib_color_maps.OrRd_r)

    plot = k3d.plot(screenshot_scale=1.0, antialias=0, environment="neutral",
                    renderer="cinematic", cinematic_bounces=4, cinematic_samples=SAMPLES,
                    background_color=-1, grid_visible=False, camera_auto_fit=False)
    plot.lighting = 1.5
    plot.colorbar_object_id = 0
    plot.cinematic_seed = 1
    plot.axes_helper = 0
    plot += volume
    plot.camera = [2.4, -2.4, 1.6, 0, 0, 0, 0, 0, 1]

    headless = k3d_remote(plot, get_headless_driver(no_headless=False, gpu=True),
                          width=WIDTH, height=HEIGHT, port=PORT)

    try:
        print("gl: %s" % (headless.get_gl_info() or {}).get("unmaskedRenderer", "?"))
        print("%dx%d, %d sampli, seed 1, environment 'neutral'\n" % (WIDTH, HEIGHT, SAMPLES))

        plot.environment_rotation = 0.0
        base = shot(headless)

        # control: nothing changed, so a pinned seed must reproduce the frame exactly. Without
        # this the other two rows could be measuring ordinary Monte Carlo noise.
        plot.environment_rotation = 0.0
        same = shot(headless)
        ok = compare("kontrola: ten sam obrot dwa razy", base, same)

        # the rotation-only path: no object changed, so ensurePrepared takes `key !== envKey`
        plot.environment_rotation = 2.0
        turned = shot(headless)
        compare("sam obrot 0.0 -> 2.0", base, turned)

        # the rebuild path: an object change sets sceneDirty, so the environment is applied
        # inside buildScene on a freshly created scene instead
        plot.environment_rotation = 4.0
        volume.color_range = [90, 900]
        rebuilt = shot(headless)
        compare("obrot 2.0 -> 4.0 razem z color_range", turned, rebuilt)

        # and back, to show the rotation is the thing that moved rather than color_range
        plot.environment_rotation = 2.0
        back = shot(headless)
        compare("obrot 4.0 -> 2.0, color_range bez zmian", rebuilt, back)

        print("")
        if not ok:
            print("  UWAGA: kontrola nie jest identyczna - seed nie trzyma, reszta niemiarodajna")
        else:
            print("  Kontrola identyczna, wiec kazda roznica ponizej pochodzi z obrotu.")
    finally:
        headless.close()


if __name__ == "__main__":
    main()
