"""Probe, not a test: does returning a big string from execute_script accumulate in the page?

Three ways of getting the same bytes out - returned, parked on window, parked and deleted - with
no renderer involved. If only the first grows, the return path owns the leak.

    python k3d/test/probe_webdriver_string_leak.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from selenium import webdriver

from k3d.headless import _relax_timeouts

CALLS = int(os.environ.get("K3D_LEAK_CALLS", "24"))
MB = int(os.environ.get("K3D_LEAK_MB", "10"))

MEM = """
if (window.gc) { window.gc(); window.gc(); }
return performance.memory.usedJSHeapSize;
"""

# One string of the requested size, built the way toDataURL builds one: a flat sequential string.
MAKE = """
var mb = arguments[0];
var parts = [];
for (var i = 0; i < mb * 16; i++) { parts.push('0123456789abcdef'.repeat(4096)); }
return parts.join('');
"""

VARIANTS = {
    # what get_screenshot does today: the whole payload crosses the wire as the script's result
    "zwraca caly string": MAKE,
    # the payload never leaves the page; only its length is returned
    "trzyma w window, zwraca dlugosc": MAKE.replace(
        "return parts.join('');",
        "window.__payload = parts.join(''); return window.__payload.length;",
    ),
    # same, then dropped before the next call, so at most one can ever be alive
    "trzyma i kasuje": MAKE.replace(
        "return parts.join('');",
        "window.__payload = parts.join('');\n"
        "var n = window.__payload.length;\n"
        "delete window.__payload;\n"
        "return n;",
    ),
}


def driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--headless=new")
    options.add_argument("--js-flags=--expose-gc")
    options.add_argument("--enable-precise-memory-info")

    return _relax_timeouts(webdriver.Chrome(options=options))


def run(label, script):
    d = driver()

    try:
        d.get("data:text/html,<title>leak</title>")
        base = d.execute_script(MEM) / 1048576.0
        samples = []

        for _ in range(CALLS):
            d.execute_script(script, MB)
            samples.append(d.execute_script(MEM) / 1048576.0)

        # over the tail, so a one-off warm-up does not read as a trend
        tail = samples[len(samples) // 3:]
        per_call = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)

        print("  %-32s start %7.1f  koniec %7.1f  %+7.2f MB/wywolanie  %s"
              % (label, base, samples[-1], per_call,
                 "ROSNIE" if per_call > MB * 0.5 else ("podejrzane" if per_call > 1.0 else "plasko")))

        return per_call
    finally:
        d.quit()


def main():
    print("  %d wywolan po %d MB, wymuszony GC przed kazdym pomiarem\n" % (CALLS, MB))

    rates = {label: run(label, script) for label, script in VARIANTS.items()}

    print("")
    first = rates["zwraca caly string"]

    if first > MB * 0.5:
        print("  Sciezka zwrotna zatrzymuje kazdy zwrocony string: %.1f MB na wywolanie" % first)
        print("  przy %d MB payloadu. To wystarcza, zeby wyjasnic awarie." % MB)
    else:
        print("  Zwracanie stringa NIE narasta - przyczyna lezy gdzie indziej.")


if __name__ == "__main__":
    main()
