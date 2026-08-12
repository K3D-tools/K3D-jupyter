"""Guards the binary layout of browser-written snapshots.

test_snapshot.py round-trips within one version, so it passes even if the encoding
changes. These assertions are deliberately written against the bytes, in Python, so they
hold whichever msgpack implementation the frontend uses.

The extension codes are msgpack-lite's preset assignments (0x11..0x1D for typed arrays)
plus 0x20, which K3D invented for its Float16Array stand-in. None of them is standardised
- MessagePack leaves 0..127 to applications - so the only contract they carry is with the
.k3d files and exported HTML already in circulation.
"""
import base64
import os
import zlib

import msgpack
import pytest

from .plot_compare import prepare

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
FIXTURE = os.path.join(FIXTURES_DIR, "snapshot_msgpack_lite.k3d")

UINT8 = 0x12
UINT32 = 0x16
FLOAT32 = 0x17
FLOAT16 = 0x20

GET_SNAPSHOT = """
const done = arguments[arguments.length - 1];
Promise.resolve(K3DInstance.getSnapshot(9)).then((bytes) => {
    let s = '';
    const a = new Uint8Array(bytes);
    for (let i = 0; i < a.length; i++) s += String.fromCharCode(a[i]);
    done(btoa(s));
}).catch((e) => done('ERROR:' + String(e)));
"""


SET_SNAPSHOT = """
const done = arguments[arguments.length - 1];
const payload = arguments[0];

try {
    K3DInstance.setSnapshot(payload);
} catch (e) {
    done({error: String(e)});
}

// Objects are created asynchronously, so wait for the list to fill before reading it.
let tries = 0;
const poll = setInterval(() => {
    const list = K3DInstance.getWorld().ObjectsListJson;
    const ids = Object.keys(list);

    if (ids.length >= 3 || ++tries > 100) {
        clearInterval(poll);
        done(ids.map((id) => {
            const o = list[id];
            const rec = {type: o.type};
            ['positions', 'voxels', 'volume'].forEach((key) => {
                if (o[key] && o[key].data) {
                    // The bundle is minified, so constructor.name is useless here; the internal
                    // class tag survives, and the stand-in is told apart by identity.
                    rec[key] = {
                        tag: Object.prototype.toString.call(o[key].data).slice(8, -1),
                        float16: o[key].data.constructor === window.Float16Array,
                        len: o[key].data.length,
                    };
                }
            });
            return rec;
        }));
    }
}, 50);
"""


def _ext_codes(raw):
    """Every extension code in a snapshot, mapped to the payload sizes carried under it."""
    data = msgpack.unpackb(zlib.decompress(raw), strict_map_key=False)
    found = {}

    def walk(node):
        if isinstance(node, msgpack.ExtType):
            found.setdefault(node.code, []).append(len(node.data))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(data)
    return data, found


def test_fixture_keeps_its_extension_codes():
    with open(FIXTURE, "rb") as f:
        data, codes = _ext_codes(f.read())

    assert sorted(data.keys()) == ["chunkList", "objects", "plot"]
    assert len(data["objects"]) == 3
    assert set(codes) == {UINT8, UINT32, FLOAT32, FLOAT16}

    # 8*8*8 halves - the stand-in stores raw uint16, so two bytes per sample
    assert 1024 in codes[FLOAT16]


def test_current_build_writes_the_same_extension_codes():
    """The migration guard: a different encoder must not shift the codes."""
    import numpy as np

    import k3d

    prepare()

    rs = np.random.RandomState(0)
    pytest.plot += k3d.points(rs.rand(32, 3).astype(np.float32), point_size=0.1)
    pytest.plot += k3d.voxels(np.ones((4, 4, 4), dtype=np.uint8))
    pytest.plot += k3d.volume(rs.rand(8, 8, 8).astype(np.float16))
    pytest.headless.sync(hold_until_refreshed=True)

    encoded = pytest.headless.browser.execute_async_script(GET_SNAPSHOT)
    assert not encoded.startswith("ERROR:"), encoded

    data, codes = _ext_codes(base64.b64decode(encoded))

    with open(FIXTURE, "rb") as f:
        _, expected = _ext_codes(f.read())

    assert set(codes) == set(expected)
    assert 1024 in codes[FLOAT16]
    assert len(data["objects"]) == 3

    while len(pytest.plot.objects) > 0:
        pytest.plot -= pytest.plot.objects[-1]
    pytest.headless.sync(hold_until_refreshed=True)


def test_current_build_reads_a_msgpack_lite_snapshot():
    """The direction users feel: a file written by an older K3D has to still open."""
    prepare()

    with open(FIXTURE, "rb") as f:
        payload = base64.b64encode(f.read()).decode()

    objects = pytest.headless.browser.execute_async_script(SET_SNAPSHOT, payload)
    assert not isinstance(objects, dict), objects

    by_type = {o["type"]: o for o in objects}
    assert sorted(by_type) == ["Points", "Volume", "Voxels"]

    # 32 points of xyz, decoded back into the same typed array the encoder started from
    assert by_type["Points"]["positions"] == {
        "tag": "Float32Array", "float16": False, "len": 96,
    }
    assert by_type["Voxels"]["voxels"] == {
        "tag": "Uint8Array", "float16": False, "len": 64,
    }
    # the stand-in is a Uint16Array holding raw halves: one element per sample of the 8x8x8 volume
    assert by_type["Volume"]["volume"] == {
        "tag": "Uint16Array", "float16": True, "len": 512,
    }

    prepare()
