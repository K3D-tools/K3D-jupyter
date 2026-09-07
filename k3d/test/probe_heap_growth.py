"""Probe, not a test: which per-frame change grows the page's JS heap.

Two axes - what the loop changes between frames, and how far along getScreenshot it goes. The
heap is read after an explicit gc(), so what is reported survived collection.

    python k3d/test/probe_heap_growth.py

K3D_HEAP_FRAMES, K3D_HEAP_SAMPLES, K3D_HEAP_WIDTH, K3D_HEAP_HEIGHT, K3D_HEAP_VOXELS and
K3D_HEAP_VARIANTS override the defaults.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import k3d
from k3d.headless import _relax_timeouts, k3d_remote

FRAMES = int(os.environ.get("K3D_HEAP_FRAMES", "16"))
SAMPLES = int(os.environ.get("K3D_HEAP_SAMPLES", "2"))
WIDTH = int(os.environ.get("K3D_HEAP_WIDTH", "3840"))
HEIGHT = int(os.environ.get("K3D_HEAP_HEIGHT", "2160"))
VOXELS = int(os.environ.get("K3D_HEAP_VOXELS", "192"))
# not 8080: a real headless session may already hold it, and this must not disturb one
PORT = int(os.environ.get("K3D_HEAP_PORT", "8099"))

ALL = {'camera': True, 'rotation': True, 'focus': True, 'colour': True}
NONE = {'camera': False, 'rotation': False, 'focus': False, 'colour': False}

VARIANTS = [
    # the screenshot axis, with nothing else moving
    ("shot:none", NONE, "none"),
    ("shot:canvas", NONE, "canvas"),
    ("shot:encode", NONE, "encode"),
    ("shot:full", NONE, "full"),
    # the property axis, at whichever screenshot depth turns out to matter
    ("prop:camera", {**NONE, 'camera': True}, "full"),
    ("prop:rotation", {**NONE, 'rotation': True}, "full"),
    ("prop:colour", {**NONE, 'colour': True}, "full"),
    ("prop:all", ALL, "full"),
]

SHOT = {
    "none": None,
    "canvas": "return K3DInstance.getScreenshot(K3DInstance.parameters.screenshotScale, 1)"
              ".then(function (d) { return d.width; });",
    "encode": "return K3DInstance.getScreenshot(K3DInstance.parameters.screenshotScale, 1)"
              ".then(function (d) { d.toDataURL(); return d.width; });",
    "full": "return K3DInstance.getScreenshot(K3DInstance.parameters.screenshotScale, 1)"
            ".then(function (d) { return d.toDataURL().split(',')[1].length; });",
}

MEM = """
if (window.gc) { window.gc(); window.gc(); }
if (!performance.memory) { return null; }
return [performance.memory.usedJSHeapSize, performance.memory.totalJSHeapSize];
"""


def expose_gc_driver():
    """A driver that lets us collect before measuring, so what we report is retention."""
    from selenium import webdriver

    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--headless=new")
    options.add_argument("--ignore-gpu-blocklist")
    options.add_argument("--enable-webgl")
    options.add_argument("--js-flags=--expose-gc")
    # without this performance.memory is quantised and cached: a 13 MB string reads as
    # +0.00 MB and every measurement built on it is a fiction
    options.add_argument("--enable-precise-memory-info")

    return _relax_timeouts(webdriver.Chrome(options=options))


def blob(n):
    """Built a slice at a time: a full meshgrid at the sizes that matter here (723 cubed is the
    user's own scan) would want several gigabytes of float32 scratch to produce 750 MB of
    float16."""
    g = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    y, x = np.meshgrid(g, g, indexing="ij")
    rxy = x ** 2 + y ** 2
    out = np.empty((n, n, n), dtype=np.float16)

    for i, z in enumerate(g):
        r2 = rxy + z * z
        out[i] = (np.exp(-r2 / 0.35) * (0.75 + 0.25 * np.cos(18.0 * x)) * 900).astype(np.float16)

    return out


def main():
    wanted = os.environ.get("K3D_HEAP_VARIANTS")
    variants = [v for v in VARIANTS if wanted is None or v[0] in wanted.split(",")]

    volume = k3d.volume(blob(VOXELS), alpha_coef=300, samples=256, light_scale=2.25,
                        color_range=[80, 900],
                        color_map=k3d.matplotlib_color_maps.OrRd_r)

    plot = k3d.plot(screenshot_scale=1.0, antialias=0, environment="neutral",
                    renderer="cinematic", cinematic_bounces=6, cinematic_samples=SAMPLES,
                    background_color=-1, grid_visible=False, camera_auto_fit=False)
    plot.lighting = 1.5
    plot.colorbar_object_id = 0
    plot.cinematic_bokeh_size = 0.1
    plot.cinematic_denoise = 2.0
    plot.cinematic_seed = 1
    plot.axes_helper = 0
    plot += volume

    headless = k3d_remote(plot, expose_gc_driver(), width=WIDTH, height=HEIGHT, port=PORT)

    try:
        print("gl: %s" % (headless.get_gl_info() or {}).get("unmaskedRenderer", "?"))
        print("%d frames per variant, %d samples, %dx%d (%.2f Mpx), volume %d^3"
              % (FRAMES, SAMPLES, WIDTH, HEIGHT, WIDTH * HEIGHT / 1e6, VOXELS))

        if headless.browser.execute_script(MEM) is None:
            print("performance.memory is unavailable - nothing to measure")
            return

        print("")
        print("  variant          first MB    last MB   MB/frame   verdict")

        for name, does, shot in variants:
            script = SHOT[shot]

            headless.sync(hold_until_refreshed=True)

            if script:
                headless.browser.execute_script(script)  # warm up outside the measurement

            used = []

            for i in range(FRAMES):
                rad = (i + 1) / FRAMES * (4.0 * math.pi)
                r = 3.0

                if does["rotation"]:
                    plot.environment_rotation = rad / 2.0
                if does["focus"]:
                    plot.cinematic_focus_distance = r * (1.0 + 0.01 * i)
                if does["camera"]:
                    plot.camera = [r * math.cos(-rad), r * math.sin(-rad), 0.7 * math.cos(rad / 2),
                                   0, 0, 0.7 * math.cos(rad / 2),
                                   -math.cos(-rad - 0.1), -math.sin(-rad - 0.1), 0]
                if does["colour"]:
                    volume.color_range = [80 + i, 900]

                headless.sync(hold_until_refreshed=True)

                if script:
                    headless.browser.execute_script(script)

                used.append(headless.browser.execute_script(MEM)[0] / 1048576.0)

            # over the tail: the first frames still carry one-off allocation
            tail = used[len(used) // 3:]
            per_frame = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
            verdict = "LEAKS" if per_frame > 1.0 else ("suspect" if per_frame > 0.25 else "flat")

            print("  %-15s %8.1f   %8.1f   %8.2f   %s"
                  % (name, used[0], used[-1], per_frame, verdict))
            sys.stdout.flush()
    finally:
        headless.close()


if __name__ == "__main__":
    main()
