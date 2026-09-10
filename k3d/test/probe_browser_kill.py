"""Probe, not a test: what exactly dies when a large volume kills the browser.

Uploading a few hundred megabytes has twice taken down every Chrome window on the machine, with
nothing in any log, no VRAM exhaustion, and not on demand - 384 cubed died where 512 survived.
So this records a timeline instead: chrome.exe PIDs and GPU state every half second, plus
Chrome's own log. Whether the PIDs go one at a time or all at once says where the cause is.

    python k3d/test/probe_browser_kill.py

GPU footprint: one K3D_KILL_VOXELS-cubed float16 volume (384 is 108 MB) plus a 720p frame.
"""
import contextlib
import csv
import os
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import k3d
from k3d.headless import _relax_timeouts, k3d_remote

VOXELS = int(os.environ.get("K3D_KILL_VOXELS", "384"))
FRAMES = int(os.environ.get("K3D_KILL_FRAMES", "6"))
PORT = int(os.environ.get("K3D_KILL_PORT", "8099"))
OUT = os.environ.get("K3D_KILL_OUT", "browser_kill")

_stop = threading.Event()


def chrome_pids():
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq chrome.exe", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []

    pids = []
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) > 1 and parts[1].isdigit():
            pids.append(int(parts[1]))

    return sorted(pids)


def gpu_state():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return [p.strip() for p in out.split(",")]
    except Exception:
        return ["", "", "", ""]


def monitor(path, note):
    """Half a second apart, so the order of deaths is visible rather than inferred."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "etap", "chrome_procesow", "util", "vram_mb", "watt", "mhz", "pidy"])

        while not _stop.is_set():
            pids = chrome_pids()
            g = gpu_state()
            w.writerow([round(time.time(), 2), note[0], len(pids)] + g
                       + [" ".join(map(str, pids))])
            f.flush()
            _stop.wait(0.5)


def blob(n):
    g = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    y, x = np.meshgrid(g, g, indexing="ij")
    rxy = x ** 2 + y ** 2
    out = np.empty((n, n, n), dtype=np.float16)

    for i, z in enumerate(g):
        out[i] = (np.exp(-(rxy + z * z) / 0.35) * 900).astype(np.float16)

    return out


def driver(log_path):
    from selenium import webdriver

    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--headless=new")
    options.add_argument("--ignore-gpu-blocklist")
    options.add_argument("--enable-webgl")
    # Chrome's own account of its last moments, which the Windows logs do not have
    options.add_argument("--enable-logging")
    options.add_argument("--v=1")
    options.add_argument(f"--log-file={log_path}")

    return _relax_timeouts(webdriver.Chrome(options=options))


def main():
    csv_path = os.path.abspath(OUT + ".csv")
    log_path = os.path.abspath(OUT + "_chrome.log")
    note = ["start"]

    t = threading.Thread(target=monitor, args=(csv_path, note), daemon=True)
    t.start()

    print("  monitor -> %s" % csv_path)
    print("  log Chrome -> %s" % log_path)
    print("  wolumen %d^3 = %.0f MB" % (VOXELS, VOXELS ** 3 * 2 / 1048576))
    print("  chrome.exe przed startem: %d procesow" % len(chrome_pids()))

    headless = None

    try:
        note[0] = "generowanie"
        data = blob(VOXELS)

        note[0] = "budowa_plota"
        volume = k3d.volume(data, alpha_coef=200, samples=128, color_range=[80, 900],
                            color_map=k3d.matplotlib_color_maps.OrRd_r)
        plot = k3d.plot(screenshot_scale=1.0, antialias=0, environment="neutral",
                        renderer="cinematic", cinematic_samples=2, background_color=-1,
                        grid_visible=False, camera_auto_fit=False)
        plot.cinematic_seed = 1
        plot.axes_helper = 0
        plot += volume
        plot.camera = [2.4, -2.4, 1.6, 0, 0, 0, 0, 0, 1]

        note[0] = "start_przegladarki"
        headless = k3d_remote(plot, driver(log_path), width=1280, height=720, port=PORT)

        note[0] = "upload_wolumenu"
        headless.sync(hold_until_refreshed=True)
        print("  [ok] wolumen dotarl, chrome.exe: %d" % len(chrome_pids()))

        note[0] = "renderowanie"
        for i in range(FRAMES):
            headless.get_screenshot(True)
            print("  klatka %d, chrome.exe: %d" % (i, len(chrome_pids())))

        print("\n  PRZEZYLO - nie udalo sie odtworzyc")
    except Exception as e:
        note[0] = "po_awarii"
        time.sleep(3)  # let the monitor catch what happens after
        print("\n  PADLO w etapie '%s': %s" % (note[0], type(e).__name__))
        print("  chrome.exe po awarii: %d procesow" % len(chrome_pids()))
    finally:
        _stop.set()
        t.join(timeout=5)

        if headless is not None:
            with contextlib.suppress(Exception):
                headless.close()

        print("\n  osi czasu w %s" % csv_path)


if __name__ == "__main__":
    main()
