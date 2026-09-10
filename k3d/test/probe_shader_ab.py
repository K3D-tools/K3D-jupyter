"""Probe, not a test: what the traced frame costs, for comparing two bundles.

Cutting the sheen and iridescence lobes took 7.2% of the fragment shader's text, but text is not
time: they were evaluated on every bounce and multiplied by zero, so the question is what the
frame costs now. This measures one bundle; the caller builds the other one and runs it again.

Two scenes, because the lobes live in specularEval and bsdfEval - surface interactions. A
volume-only scene is what the author's animation actually renders; the one with geometry is where
the effect should be largest, and the difference between them says whether the win generalises.

    K3D_AB_LABEL=przed python k3d/test/probe_shader_ab.py

Pinned seed and a fixed budget, so the work is identical across runs and the only variable is the
shader. GPU footprint: 1280x720 and a 192-cubed float16 volume, about 14 MB - deliberately small.
"""
import os
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import k3d
from k3d.headless import get_headless_driver, k3d_remote

LABEL = os.environ.get("K3D_AB_LABEL", "?")
REPEATS = int(os.environ.get("K3D_AB_REPEATS", "7"))
SAMPLES = int(os.environ.get("K3D_AB_SAMPLES", "64"))
WIDTH = int(os.environ.get("K3D_AB_WIDTH", "1280"))
HEIGHT = int(os.environ.get("K3D_AB_HEIGHT", "720"))
VOXELS = int(os.environ.get("K3D_AB_VOXELS", "192"))
PORT = int(os.environ.get("K3D_AB_PORT", "8095"))


def power():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().split("\n")[0])
    except Exception:
        # None, not a nan: a reading that failed is a missing sample,
        # and saying so beats leaning on nan != nan to filter it
        return None


def blob(n):
    g = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    y, x = np.meshgrid(g, g, indexing="ij")
    rxy = x ** 2 + y ** 2
    out = np.empty((n, n, n), dtype=np.float16)

    for i, z in enumerate(g):
        out[i] = (np.exp(-(rxy + z * z) / 0.35)
                  * (0.75 + 0.25 * np.cos(18.0 * x)) * 900).astype(np.float16)

    return out


def glossy_mesh():
    """Surfaces are where specularEval and bsdfEval run, so the lobes cost most here."""
    rings, w = 60, 30
    az = np.linspace(0, 2 * np.pi, rings, dtype=np.float32)
    el = np.linspace(0, np.pi, w, dtype=np.float32)
    verts, faces = [], []

    for a in az:
        for b in el:
            verts.append([1.6 * np.sin(b) * np.cos(a),
                          1.6 * np.sin(b) * np.sin(a),
                          1.6 * np.cos(b)])

    for i in range(rings - 1):
        for j in range(w - 1):
            lo, hi = i * w + j, (i + 1) * w + j
            faces += [[lo, hi, lo + 1], [hi, hi + 1, lo + 1]]

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.uint32)


def timed(headless, plot, label):
    headless.sync(hold_until_refreshed=True)
    headless.get_screenshot(True)  # warm up: the first frame pays for compiling the program

    times, watts = [], []

    for _ in range(REPEATS):
        t0 = time.perf_counter()
        headless.get_screenshot(True)
        times.append(time.perf_counter() - t0)
        watts.append(power())

    drawn = [w for w in watts if w is not None]

    print("  %-22s mediana %6.2f s   min %6.2f   max %6.2f   %s"
          % (label, statistics.median(times), min(times), max(times),
             "%5.1f W" % statistics.mean(drawn) if drawn else "   ? W"))

    return statistics.median(times)


def main():
    print("  bundle: %s   %dx%d, %d sampli, %d powtorzen\n"
          % (LABEL, WIDTH, HEIGHT, SAMPLES, REPEATS))

    volume = k3d.volume(blob(VOXELS), alpha_coef=300, samples=256, light_scale=2.25,
                        color_range=[80, 900], color_map=k3d.matplotlib_color_maps.OrRd_r)

    plot = k3d.plot(screenshot_scale=1.0, antialias=0, environment="studio",
                    renderer="cinematic", cinematic_bounces=6, cinematic_samples=SAMPLES,
                    background_color=-1, grid_visible=False, camera_auto_fit=False)
    plot.lighting = 1.5
    plot.colorbar_object_id = 0
    plot.cinematic_seed = 1
    plot.axes_helper = 0
    plot.camera = [3.4, -3.4, 2.2, 0, 0, 0, 0, 0, 1]
    plot += volume

    headless = k3d_remote(plot, get_headless_driver(gpu=True), width=WIDTH, height=HEIGHT,
                          port=PORT)

    try:
        print("  gl: %s\n" % (headless.get_gl_info() or {}).get("unmaskedRenderer", "?")[:70])

        vol_only = timed(headless, plot, "tylko wolumen")

        verts, faces = glossy_mesh()
        mesh = k3d.mesh(verts, faces, color=0xc0c0c0, flat_shading=False)
        mesh.roughness = 0.15
        mesh.metalness = 0.9
        plot += mesh

        with_mesh = timed(headless, plot, "wolumen + siatka")

        print("")
        print("  %s|%.4f|%.4f" % (LABEL, vol_only, with_mesh))
    finally:
        headless.close()


if __name__ == "__main__":
    main()
