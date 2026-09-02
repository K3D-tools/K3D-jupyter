import re

import numpy as np

import k3d
from k3d._version import __version__

UNPKG = re.compile(r"https://unpkg\.com/k3d@([^/]+)/dist/standalone\.js")

POINTS = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)


def _snapshot(snapshot_type):
    plot = k3d.plot()
    plot += k3d.points(POINTS)
    plot.snapshot_type = snapshot_type

    return plot.get_snapshot()


def test_online_and_inline_snapshots_point_at_this_k3d_on_unpkg():
    # Regression: the substitution used _view_module_version, which since 3.0.0 is anywidget's
    # range, so both templates asked unpkg for k3d@~0.9.* and no snapshot could load the bundle.
    for snapshot_type in ("online", "inline"):
        urls = set(UNPKG.findall(_snapshot(snapshot_type)))

        assert urls == {__version__}, "%s snapshot requested %s" % (snapshot_type, urls)


def test_full_snapshot_needs_no_cdn():
    # `full` inlines the bundle, so it must not reference unpkg at all
    assert not UNPKG.search(_snapshot("full"))
