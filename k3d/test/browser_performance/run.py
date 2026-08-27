"""Browser performance harness: one command, one page, one CSV.

Not a pytest test and deliberately not part of CI - it needs a real GPU and a visible window, and
CI runners have neither. What it does have is the visual suite's own scenes: `corpus.py` replays
every visual test against stubs and writes what it built as .k3d files, so the catalogue of scenes
maintains itself.

    python run.py                       # generate what is missing, then serve on a free port
    python run.py --bundles local 2.18.0
    python run.py --regenerate          # rebuild the corpus first
    python run.py --report results/perf_x.csv  # every bundle in one run vs the oldest
    python run.py --compare a.csv b.csv # ratio table from two separate runs

Then open the URL it prints, tick the bundles and press Start. Every finished scene is posted back
and appended to the CSV immediately, so a run that dies half way still leaves half the answers.

A scene is measured with as many copies of itself as it takes to make a frame cost the target. The
count is found with a vsync-free probe - forced renders timed against a one-pixel readback, median
of three batches - scaled towards the target one step at a time: n <- n * aim / cost(n). Frame
timing cannot find it, because vsync pins the frame to the refresh period until the object's own
cost passes it, so a reading on the floor carries no information. The aim is deliberately above the
target and anything from the target to three times it is accepted: landing short leaves the frame on
the floor measuring the refresh rate, landing long only makes it slower to measure. Every step is
bounded to eightfold, so a bad reading costs a step rather than the run, and the whole search is
written to the CSV as probePath.

Copies are scene-graph clones sharing the original's geometry, so making one costs microseconds and
no memory. Loading the object's JSON again instead - which is what the page's `duplicate` knob still
does - rebuilds the mesh per copy: on menger_sponge that was 1.4 s and 167 MB of GPU buffers each,
and 64 copies then measured video memory paging rather than k3d.

Two rules make the numbers comparable. Bundles are measured oldest first with `local` last, and the
first of them is the only one that searches: its copy count is imposed on every later bundle, so one
scene is one workload no matter who renders it. A bundle faster than the reference can then finish
inside a refresh period, where the frame time says nothing - those rows are marked under-resolved
instead of being reported as measurements, and `probeMs` in the CSV is the unquantised number to
compare in that case. And every bundle that has the renderer modes is measured in `simple` and
`advanced` (never `cinematic`, which is not a real-time path); a bundle older than the modes has no
setRenderer, is measured once, and that one row is compared against both.
"""
import argparse
import contextlib
import csv
import datetime
import glob
import json
import os
import socket
import statistics
import sys
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

# runnable both as `python run.py` from this directory and as a module from the repository root;
# corpus.py imports the test package by name, so the root has to be importable either way
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    from . import bundles, corpus
except ImportError:
    import bundles
    import corpus

bundles_module = bundles
SCENES_DIR = os.path.join(HERE, 'scenes')
RESULTS_DIR = os.path.join(HERE, 'results')
STATIC_DIR = os.path.join(HERE, 'static')
BUNDLES_DIR = os.path.join(HERE, 'bundles')

# Fixed column order, and wider than the three numbers the eye needs: a month from now the only
# way to know whether two rows are comparable is the metadata that came with them.
COLUMNS = (
    'timestamp', 'bundle', 'scene', 'renderer', 'rendererRequested', 'supportsModes',
    'sets', 'setsFrom', 'objects', 'objectsAt1', 'duplicate', 'menu',
    'medianMs', 'p95Ms', 'workMs', 'emptyMs', 'dominance', 'underResolved', 'aborted',
    'probeMs', 'probeSpreadPct', 'probeAt1Ms', 'perCopyMs', 'probeLeverSets',
    'probeStepsUsed', 'probePath', 'correctionSkipped',
    'corrected',
    'drawsPerFrame', 'framesMeasured', 'programs',
    'loadMs', 'bundleMs', 'fetchMs',
    'sceneBytes', 'fileBytes', 'gpuBytesAt1', 'trianglesAt1', 'geometriesAt1',
    'bufferW', 'bufferH', 'devicePixelRatio', 'samples',
    'coverageAt1', 'coverageFinal', 'coverageSaturated', 'imageAt1', 'imageFinal',
    'transformIgnored', 'capped',
    'gpu', 'error',
)


def _port_holder(port):
    """Who has the port, in words - a stack trace helps nobody here."""
    try:
        import subprocess

        out = subprocess.run(['docker', 'ps', '--format', '{{.Names}} {{.Ports}}'],
                             capture_output=True, text=True, timeout=5).stdout

        for line in out.splitlines():
            if ':%d->' % port in line:
                return 'docker container %s' % line.split()[0]
    except Exception:  # diagnostics must not fail the run
        pass

    return 'another process'


def _free(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            probe.bind(('127.0.0.1', port))
        except OSError:
            return False

    return True


def _pick_port():
    """Let the system name a free port.

    A fixed default would be a guess at what nobody else is using, and 8888 in particular is
    Jupyter's - whoever runs this most likely has a notebook on it already. The number is printed
    and the browser is opened with it, so nobody has to know it.
    """
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))

        return probe.getsockname()[1]


def ensure_corpus(regenerate=False):
    manifest_path = os.path.join(SCENES_DIR, 'manifest.json')

    if os.path.isfile(manifest_path) and not regenerate:
        with open(manifest_path, encoding='utf-8') as handle:
            return json.load(handle)['scenes'], None

    # the visual tests reach for their assets by relative path, so the corpus has to be built from
    # the package directory - the same working directory CI uses for the suite
    previous = os.getcwd()
    os.chdir(os.path.dirname(os.path.dirname(HERE)))

    try:
        manifest, report = corpus.generate(SCENES_DIR)
    finally:
        os.chdir(previous)

    return manifest, report


def ensure_bundles(versions):
    ready = []

    for version in versions:
        try:
            bundles.ensure(version)
            ready.append(version)
        except Exception as exc:  # a missing release is a message, not a crash
            print('  %-10s failed: %s' % (version, exc))

    return ready


def make_app(manifest, csv_path):
    from flask import Flask, jsonify, request, send_from_directory

    app = Flask(__name__, static_folder=None)
    app.logger.disabled = True

    @app.route('/')
    def index():
        return send_from_directory(STATIC_DIR, 'index.html')

    @app.route('/static/<path:name>')
    def static_file(name):
        return send_from_directory(STATIC_DIR, name)

    @app.route('/scenes/<path:name>')
    def scene_file(name):
        return send_from_directory(SCENES_DIR, name)

    @app.route('/bundles/<path:name>')
    def bundle_file(name):
        return send_from_directory(BUNDLES_DIR, name)

    @app.route('/api/config')
    def config():
        return jsonify({
            'scenes': manifest,
            'bundles': bundles.available(),
            'csv': os.path.relpath(csv_path, HERE),
        })

    @app.route('/api/row', methods=['POST'])
    def row():
        payload = request.get_json(force=True) or {}
        payload['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
        fresh = not os.path.isfile(csv_path)

        with open(csv_path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction='ignore')

            if fresh:
                writer.writeheader()

            writer.writerow(payload)

        notes = [payload.get('error'), payload.get('capped')]

        if payload.get('underResolved'):
            notes.append('under-resolved')

        if payload.get('aborted'):
            notes.append('aborted')

        print('  %-36s %-10s %-8s sets=%-5s median=%-7s probe=%-7s %s' % (
            str(payload.get('scene'))[:36], payload.get('bundle'),
            payload.get('renderer') or '-', payload.get('sets'),
            ('%.2f' % payload['medianMs']) if payload.get('medianMs') else '-',
            ('%.2f' % payload['probeMs']) if payload.get('probeMs') else '-',
            ' '.join(n for n in notes if n)))

        return jsonify({'ok': True})

    @app.route('/api/done', methods=['POST'])
    def done():
        print()
        print('written to %s' % csv_path)

        return jsonify({'ok': True})

    return app


# a frame-time difference smaller than one refresh period is quantisation, not performance
_FRAME_FLOOR_MS = 7.0


def _rows_by_scene(rows):
    """{scene: {(bundle, renderer): row}} for rows that carry a measurement."""
    out = {}

    for row in rows:
        if row.get('error') or not row.get('medianMs') or row['scene'] == '__empty__':
            continue

        out.setdefault(row['scene'], {})[(row['bundle'], row.get('renderer') or '')] = row

    return out


def _ratio(a, b, field):
    try:
        left = float(a.get(field) or 0)
        right = float(b.get(field) or 0)
    except ValueError:
        return None

    return (right / left) if left > 0 else None


def _same_pipeline(a, mode):
    """Whether both sides of a pair went down the same rendering path.

    2.18.0 has no renderer modes at all, so pairing its one path against `advanced` compares two
    different pipelines. Worth keeping, worth keeping apart.
    """
    return mode == (a.get('renderer') or '') or (
        not (a.get('renderer') or '') and mode in ('', 'simple'))


def _number(row, field):
    """A CSV cell as a float, or None - every numeric column can be blank or absent."""
    try:
        value = float(row.get(field) or 0)
    except (TypeError, ValueError):
        return None

    return value if value > 0 else None


def _load_floors(rows):
    """Each bundle's fixed load cost, estimated as the lightest scene it managed.

    loadMs pays for the K3D instance, the renderer and the first shader compile before it pays for
    a single object, and that part does not scale with the scene. The smallest load in the run is
    the closest thing to that floor the data contains, so subtracting it leaves roughly the part
    the scene is answerable for. It is an estimate from one sample, not a measurement.
    """
    floors = {}

    for row in rows:
        if row.get('error'):
            continue

        value = _number(row, 'loadMs')

        if value is None:
            continue

        bundle = row['bundle']

        if bundle not in floors or value < floors[bundle]:
            floors[bundle] = value

    return floors


def _load_pairs(rows, base, floors):
    """Load-time pairs.

    Two things set these apart from the frame tables. They do not require matching copy counts:
    loadMs is one CreateK3DAndLoadBinarySnapshot, one copy of the scene, whatever the copy search
    settled on afterwards. And they do not require a frame measurement at all - the load happens
    before the first frame is drawn, so a scene that loaded but could not be timed still has a
    valid loadMs, which is why this groups the rows itself instead of reusing _rows_by_scene.
    """
    grouped = {}

    for row in rows:
        if row.get('error') or row['scene'] == '__empty__':
            continue

        grouped.setdefault(row['scene'], {})[(row['bundle'], row.get('renderer') or '')] = row

    out = []

    for scene, per_scene in grouped.items():
        left = {mode: row for (bundle, mode), row in per_scene.items() if bundle == base}

        if not left:
            continue

        for (bundle, mode), b in sorted(per_scene.items()):
            if bundle == base:
                continue

            a = left.get(mode) or (next(iter(left.values())) if len(left) == 1 else None)

            if a is None:
                continue

            a_load, b_load = _number(a, 'loadMs'), _number(b, 'loadMs')

            if a_load is None or b_load is None:
                continue

            out.append({
                'scene': scene, 'bundle': bundle, 'mode': mode,
                'a_load': a_load, 'b_load': b_load, 'load': b_load / a_load,
                'delta': b_load - a_load,
                'spread': None, 'same_pipeline': _same_pipeline(a, mode), 'flags': _flags(a, b),
            })

    return out


def _load_table(subset, title, note):
    if not subset:
        return

    print(title)
    print(note)
    print('%-38s %-15s %-8s %8s %8s %8s %7s  %s' % (
        'scene', 'bundle', 'renderer', 'A ms', 'B ms', 'B-A ms', 'B/A', 'note'))

    # sorted by the absolute regression, which is the robust one here: a ratio on a scene whose
    # load is mostly the fixed cost says more about that cost than about the scene
    for p in sorted(subset, key=lambda p: -p['delta']):
        print('%-38s %-15s %-8s %8.1f %8.1f %8.1f %7.2f  %s' % (
            p['scene'][:38], p['bundle'], p['mode'] or '-', p['a_load'], p['b_load'],
            p['delta'], p['load'], p['flags']))

    values = sorted(p['load'] for p in subset)
    deltas = sorted(p['delta'] for p in subset)
    print()
    print('  n=%-4d median B/A %.3f   slower by >10%%: %d   by >25%%: %d   faster: %d'
          % (len(values), values[len(values) // 2], sum(1 for r in values if r > 1.10),
             sum(1 for r in values if r > 1.25), sum(1 for r in values if r < 0.95)))
    print('  median B-A %+.1f ms   worst %+.1f ms   best %+.1f ms'
          % (deltas[len(deltas) // 2], deltas[-1], deltas[0]))

    print('  one cold sample per row: it pays JIT warm-up, the first shader compile and'
          ' whatever the')
    print('  VRAM state happened to be - so read the medians, not any single row')
    print()


def _flags(a, b):
    out = []

    for side, row in (('A', a), ('B', b)):
        if str(row.get('underResolved')).lower() == 'true':
            out.append(side + ' at the vsync floor')

        if str(row.get('aborted')).lower() == 'true':
            out.append(side + ' aborted')

        if row.get('capped'):
            out.append(side + ' ' + row['capped'])

    return ', '.join(out)


def _newest_run(path=None):
    """The run to report on: what was asked for, or the newest one in results/."""
    if path and os.path.isfile(path):
        return path

    directory = path if path and os.path.isdir(path) else RESULTS_DIR
    found = sorted(glob.glob(os.path.join(directory, 'perf_*.csv')), key=os.path.getmtime)

    return found[-1] if found else None


def _scene_types():
    """{scene file: [k3d object types]} for the corpus.

    Taken from the manifest when it has them. Older manifests do not, so the snapshots are read
    instead - a .k3d is zlib around msgpack, and every object in it carries its own `type`, which
    is a far better grouping than what the test happened to be named.
    """
    out = {}
    manifest_path = os.path.join(SCENES_DIR, 'manifest.json')

    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding='utf-8') as handle:
            for entry in json.load(handle).get('scenes', []):
                if entry.get('types'):
                    out[entry['file']] = entry['types']

    if out:
        return out

    try:
        import zlib

        import msgpack
    except ImportError:
        return out

    for path in glob.glob(os.path.join(SCENES_DIR, '*.k3d')):
        try:
            with open(path, 'rb') as handle:
                data = msgpack.unpackb(zlib.decompress(handle.read()), raw=False)
        except Exception:  # a scene we cannot read is simply ungrouped
            continue

        types = sorted({o.get('type') for o in data.get('objects', []) if o.get('type')})

        if types:
            out[os.path.basename(path)] = types

    return out


def _family(scene, types_by_scene):
    types = types_by_scene.get(scene) or types_by_scene.get(scene + '.k3d')

    return '+'.join(types) if types else '(unknown)'


def _bands(values):
    """How many of these ratios fall in each band, faster to slower."""
    return [
        ('faster', sum(1 for v in values if v < 0.95)),
        ('parity', sum(1 for v in values if 0.95 <= v < 1.05)),
        ('5-10%', sum(1 for v in values if 1.05 <= v < 1.10)),
        ('10-25%', sum(1 for v in values if 1.10 <= v < 1.25)),
        ('25%+', sum(1 for v in values if v >= 1.25)),
    ]


def _by_family(subset, types_by_scene, spread_limit, field='probe',
               title='=== by object type (probe ratios, repeatable pairs only) ==='):
    """The like-for-like ratios grouped by what the scene actually draws."""
    if not subset:
        return

    families = {}

    for p in subset:
        if p['spread'] is not None and p['spread'] > spread_limit:
            continue

        if p.get(field) is None:
            continue

        families.setdefault(_family(p['scene'], types_by_scene), []).append(p)

    if not families:
        return

    print(title)
    print('  %-24s %4s %8s   %6s %6s %6s %6s %6s   %s'
          % ('object type', 'n', 'median', 'faster', 'parity', '5-10%', '10-25%', '25%+',
             'worst scene'))

    order = sorted(families.items(),
                   key=lambda kv: -statistics.median([p[field] for p in kv[1]]))

    for family, group in order:
        values = sorted(p[field] for p in group)
        worst = max(group, key=lambda p: p[field])
        counts = dict(_bands(values))
        print('  %-24s %4d %8.3f   %6d %6d %6d %6d %6d   %s (%.2f)'
              % (family, len(values), statistics.median(values), counts['faster'],
                 counts['parity'], counts['5-10%'], counts['10-25%'], counts['25%+'],
                 worst['scene'].replace('.k3d', '')[:30], worst[field]))

    print()


def report(path):
    """One run, every bundle in it, compared against the oldest.

    This is the shape the runner actually writes - all bundles in one file - and it is the reason
    `--compare` is not enough on its own. Two numbers per pair: the frame time, which is what
    someone using k3d feels but is quantised by vsync, and the probe, which is not quantised and
    therefore the one that resolves a small regression. They are measured at the same copy count
    by construction, so no scaling is needed to compare them.
    """
    chosen = _newest_run(path)

    if chosen is None:
        print('no run to report on in %s' % (path or RESULTS_DIR))

        return

    print('run: %s' % chosen)

    with open(chosen, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))

    by_scene = _rows_by_scene(rows)
    bundles = bundles_module.order({row['bundle'] for row in rows})

    if len(bundles) < 2:
        print('only %s in this file - nothing to compare against'
              % (', '.join(bundles) or 'nothing'))

        return

    base = bundles[0]
    print('baseline: %s     compared: %s' % (base, ', '.join(bundles[1:])))
    print()
    # Bundles are measured in a fixed order, every scene, and that order biases the result.
    # Measured with the same bytes under two names: the bundle in the SECOND slot came out 15%
    # faster than the third, fourth and fifth on every scene whose shading samples an
    # environment map, while scenes drawn with K3D's own shaders were flat to a tenth of a
    # percent. Third, fourth and fifth agree with each other, so it is the second slot that is
    # special, not a drift. Until the runner varies the order, read a ratio against the
    # baseline as carrying that bias, and settle any bundle-to-bundle question by measuring the
    # same bundle twice under two names in the same run.
    print('  NOTE: the second bundle measured in each scene is favoured - up to 15% on scenes')
    print('  that sample an environment map. Add the same bundle twice under two names to see')
    print('  how much of a difference below is that, and not the code.')
    print()

    pairs = []

    for scene, per_scene in by_scene.items():
        left = {mode: row for (bundle, mode), row in per_scene.items() if bundle == base}

        if not left:
            continue

        for (bundle, mode), b in sorted(per_scene.items()):
            if bundle == base:
                continue

            a = left.get(mode) or (next(iter(left.values())) if len(left) == 1 else None)

            if a is None or a.get('sets') != b.get('sets'):
                continue

            # A frame-time ratio is only a ratio when both sides are clear of the refresh period.
            # The frame time is quantised: two versions a single quantum apart - 13.9 ms against
            # 7.0 - give work of 6.9 against 0.1 and a "ratio" of 34, or of zero the other way
            # round. Measured on a real run, the worst offenders that way were line_thick_
            # clipping_plane at 34.00 and line_simple at 0.00, both of which the probe put within
            # 7% of parity. So the column is left empty unless both sides carry at least a full
            # refresh period of work.
            def resolved(row):
                work = row.get('workMs')

                try:
                    return work is not None and float(work) >= _FRAME_FLOOR_MS
                except (TypeError, ValueError):
                    return False

            frame = _ratio(a, b, 'workMs') if (resolved(a) and resolved(b)) else None

            same_pipeline = _same_pipeline(a, mode)

            spreads = []

            for row in (a, b):
                # a row written by an older run has no spread column, and a pair with only
                # one side reporting one is still worth comparing
                with contextlib.suppress(ValueError):
                    spreads.append(float(row.get('probeSpreadPct') or 0))

            pairs.append({
                'scene': scene, 'bundle': bundle, 'mode': mode, 'sets': a.get('sets'),
                'frame': frame, 'probe': _ratio(a, b, 'probeMs'),
                'a_probe': a.get('probeMs'), 'b_probe': b.get('probeMs'),
                'spread': max(spreads) if spreads else None,
                'same_pipeline': same_pipeline, 'flags': _flags(a, b),
            })

    ranked = sorted((p for p in pairs if p['probe'] is not None),
                    key=lambda p: -p['probe'])

    def table(subset, title, note):
        if not subset:
            return

        print(title)

        if note:
            print(note)

        print('%-38s %-9s %-8s %6s %8s %8s %7s %7s  %s' % (
            'scene', 'bundle', 'renderer', 'copies', 'A ms', 'B ms', 'B/A', 'frame', 'note'))
        print('%-38s %-9s %-8s %6s %8s %8s %7s %7s' % (
            '', '', '', '', 'per', 'render', 'ratio', 'ratio'))

        for p in subset:
            print('%-38s %-9s %-8s %6s %8.2f %8.2f %7.2f %7s  %s' % (
                p['scene'][:38], p['bundle'], p['mode'] or '-', p['sets'],
                float(p['a_probe']), float(p['b_probe']), p['probe'],
                '-' if p['frame'] is None else '%.2f' % p['frame'], p['flags']))

        group = sorted(p['probe'] for p in subset)
        frames = [p['frame'] for p in subset if p['frame'] is not None]
        print()
        print('  n=%-4d median probe %.3f   slower by >10%%: %d   by >25%%: %d   faster: %d'
              % (len(group), group[len(group) // 2], sum(1 for r in group if r > 1.10),
                 sum(1 for r in group if r > 1.25), sum(1 for r in group if r < 0.95)))
        print('  frame time resolved on both sides in %d of %d pairs' % (len(frames), len(subset)))
        print('  frame ratios are quantised to the refresh period - one quantum apart already'
              ' reads as 2x, so the probe column is the finer of the two')
        print()

    table([p for p in ranked if p['same_pipeline']],
          '=== like for like ===',
          '  the same rendering path on both sides, biggest regression first')

    _by_family([p for p in ranked if p['same_pipeline']], _scene_types(), 10.0)

    table([p for p in ranked if not p['same_pipeline']],
          '=== a different pipeline ===',
          '  %s has no renderer modes, so these compare its single path against a mode that did\n'
          '  not exist then - a ratio here is the cost of the mode, not a regression' % base)

    floors = _load_floors(rows)
    load = _load_pairs(rows, base, floors)

    if load:
        print('=== object load ===')
        print('  loadMs is one CreateK3DAndLoadBinarySnapshot resolved: the snapshot decoded,'
              ' every object')
        print('  built, its buffers and textures uploaded, the first shaders compiled - for ONE'
              ' copy of the')
        print('  scene, not the copies the frame tables measure.')
        print('  lil-gui building IS inside it, whatever the menu column says: that column'
              ' records the')
        print('  state after the load, and every snapshot carries menuVisibility true, so'
              ' initializeGUI')
        print('  and a per-object controller build run within the measured call. The two'
              ' bundles do')
        print('  not build the same set of controls, so part of every difference below is'
              ' lil-gui.')
        print('  No ms-per-byte figure here: neither recorded size is the payload this'
              ' measures. fileBytes')
        print('  is the compressed file - menger_sponge is 31 kB on disk against 531 kB of'
              ' arrays - and')
        print('  sceneBytes walks the objects only, missing an environment texture in the plot'
              ' parameters')
        print('  (4 kB of scene, 865 kB of file). The uncompressed total is recorded nowhere.')
        print('  the lightest load each bundle managed, as a guide to how much of a small'
              ' number is')
        print('  fixed cost rather than the scene: %s'
              % ('   '.join('%s %.0f ms' % (name, floors[name])
                            for name in bundles if name in floors)))
        print()

        _load_table([p for p in load if p['same_pipeline']],
                    '  --- like for like ---',
                    '  the same rendering path on both sides, worst first')

        _by_family([p for p in load if p['same_pipeline']], _scene_types(), 10.0, field='load',
                   title='  --- load by object type (B/A, fixed cost included) ---')

        _load_table([p for p in load if not p['same_pipeline']],
                    '  --- a different pipeline ---',
                    '  a mode that did not exist in %s - the number is the mode\'s load cost,'
                    ' not a regression' % base)

    # what a mode costs inside one bundle, which is a fair question the pairs above cannot answer
    for bundle in bundles[1:]:
        modes = sorted({mode for scene in by_scene for (bn, mode) in by_scene[scene] if bn == bundle
                        and mode})

        if len(modes) < 2:
            continue

        base_mode = 'simple' if 'simple' in modes else modes[0]

        for mode in modes:
            if mode == base_mode:
                continue

            costs = []

            for per_scene in by_scene.values():
                one = per_scene.get((bundle, base_mode))
                other = per_scene.get((bundle, mode))

                if one and other and one.get('sets') == other.get('sets'):
                    value = _ratio(one, other, 'probeMs')

                    if value:
                        costs.append(value)

            if costs:
                costs.sort()
                print('inside %s: %s costs %.2fx %s (median of %d scenes, up to %.2fx)'
                      % (bundle, mode, costs[len(costs) // 2], base_mode, len(costs), costs[-1]))

    skipped = len(pairs) - len(ranked)

    if skipped:
        print('pairs without a probe on both sides: %d' % skipped)


def label_of(bundle, mode):
    return bundle + ('/' + mode if mode else '')


def compare(first, second):
    """Ratio table from two finished runs.

    A pair is only a comparison if both sides measured the same scene with the same number of
    copies - the runner arranges exactly that, and a mismatch here means the two CSVs come from
    different runs, so it is reported and skipped rather than divided. Renderer modes are matched
    by name; a bundle that predates the modes has one nameless row, and that row is compared
    against each of the other side's modes.
    """
    def load(path):
        rows = {}
        collisions = []

        with open(path, newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                if row.get('error') or not row.get('workMs') or row['scene'] == '__empty__':
                    continue

                per_scene = rows.setdefault(row['scene'], {})
                mode = row.get('renderer') or ''

                # setRenderer is allowed to refuse a mode the machine cannot do, and then two
                # requested modes report the same effective one - a silent overwrite here would
                # hide that the run measured one path twice
                if mode in per_scene:
                    collisions.append('%s [%s] requested as %s and %s'
                                      % (row['scene'], mode or '-',
                                         per_scene[mode].get('rendererRequested') or '?',
                                         row.get('rendererRequested') or '?'))

                per_scene[mode] = row

        for line in collisions:
            print('duplicate in %s: %s' % (os.path.basename(path), line))

        return rows

    left = load(first)
    right = load(second)
    pairs = []
    skipped = []

    for scene in sorted(set(left) & set(right)):
        for mode, b in sorted(right[scene].items()):
            a = left[scene].get(mode)

            if a is None and len(left[scene]) == 1:
                a = next(iter(left[scene].values()))

            if a is None:
                skipped.append((scene, mode, 'no matching mode'))
                continue

            if a.get('sets') != b.get('sets'):
                skipped.append((scene, mode, 'sets %s vs %s' % (a.get('sets'), b.get('sets'))))
                continue

            a_ms = float(a['workMs'])
            b_ms = float(b['workMs'])

            if a_ms <= 0:
                skipped.append((scene, mode, 'A at the refresh floor'))
                continue

            flags = []

            for side, row in (('A', a), ('B', b)):
                try:
                    spread = float(row.get('probeSpreadPct') or 0)
                except ValueError:
                    spread = 0

                if spread > 15:
                    flags.append('%s probe +-%.0f%%' % (side, spread))

                if str(row.get('underResolved')).lower() == 'true':
                    flags.append(side + ' under-resolved')

                if str(row.get('aborted')).lower() == 'true':
                    flags.append(side + ' aborted')

                if row.get('capped'):
                    flags.append(side + ' ' + row['capped'])

            # the frame time is quantised by vsync; the probe is not, so it is the finer of the
            # two whenever both sides have it - a few percent shows up there and nowhere else
            probe = None

            if a.get('probeMs') and b.get('probeMs') and float(a['probeMs']) > 0:
                probe = float(b['probeMs']) / float(a['probeMs'])

            pairs.append((b_ms / a_ms, scene, mode, a.get('sets'), a_ms, b_ms, probe,
                          ', '.join(flags)))

    pairs.sort(reverse=True)

    print('%-38s %-9s %6s %9s %9s %7s %8s  %s'
          % ('scene', 'renderer', 'sets', 'A work', 'B work', 'B/A', 'probe', 'note'))

    for ratio, scene, mode, sets, a_ms, b_ms, probe, note in pairs:
        print('%-38s %-9s %6s %9.2f %9.2f %7.2f %8s  %s'
              % (scene[:38], mode or '-', sets, a_ms, b_ms, ratio,
                 '-' if probe is None else '%.2f' % probe, note))

    print()
    print('pairs compared: %d' % len(pairs))

    if pairs:
        median = sorted(p[0] for p in pairs)[len(pairs) // 2]
        print('median B/A: %.2f   (>1 means B is slower)' % median)

    for scene, mode, why in skipped:
        print('skipped: %s [%s] - %s' % (scene, mode or '-', why))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--port', type=int, default=0,
                        help='0 (default) - let the system choose a free port')
    parser.add_argument('--bundles', nargs='*', default=['local'])
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--compare', nargs=2, metavar=('A.csv', 'B.csv'),
                        help='two finished runs, one bundle each')
    parser.add_argument('--report', nargs='?', const='', metavar='RUN.csv',
                        help='one finished run - every bundle in it against the oldest; '
                             'with no argument, the newest run in results/')
    args = parser.parse_args()

    if args.report is not None:
        report(args.report)

        return 0

    if args.compare:
        compare(*args.compare)

        return 0

    print('scene corpus...')
    # not `report`: that is the name of the command below, and shadowing it here made --report
    # fail with an UnboundLocalError
    manifest, corpus_report = ensure_corpus(args.regenerate)

    if corpus_report is not None:
        skipped = [row for row in corpus_report if row[2] != 'ok']

        for module_name, label, status, detail in skipped:
            print('  skipped: %s::%s (%s: %s)' % (module_name, label, status, detail))

    print('  %d scenes, %.1f MB' % (len(manifest), sum(s['bytes'] for s in manifest) / 1e6))

    print('bundles...')
    ready = ensure_bundles(args.bundles)

    for version in ready:
        print('  %s' % version)

    if not ready:
        print('nothing to measure without a bundle')

        return 1

    port = args.port or _pick_port()

    if args.port and not _free(args.port):
        print('port %d is taken (%s) - free it, or drop --port and let the system choose'
              % (args.port, _port_holder(args.port)))

        return 1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'perf_%s.csv'
                            % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    url = 'http://127.0.0.1:%d/' % port

    print()
    print('open %s, tick the bundles and press Start' % url)
    print('the window has to stay visible - a background tab is given no frames')
    print()

    if not args.no_browser:
        webbrowser.open(url)

    make_app(manifest, csv_path).run(host='127.0.0.1', port=port, debug=False,
                                     use_reloader=False, threaded=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
