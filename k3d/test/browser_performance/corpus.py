"""Build the .k3d corpus from the visual test suite, without a browser.

The visual tests are already the scene catalogue: every object type, every parameter combination
worth looking at, written and maintained by people who care. They rest on three things - the
`pytest.plot` and `pytest.headless` fixtures and `plot_compare.compare()`, which renders and
asserts. Replace those three with stubs and the same test bodies become a scene generator that
needs no browser, no chromedriver and no docker.

Nothing in the k3d package is modified: the stubs are installed on the pytest module and
plot_compare.compare is rebound here, before the test modules are imported, so their
`from .plot_compare import compare` picks up the replacement.

    python -m k3d.test.browser_performance.corpus [out_dir]
"""
import importlib
import json
import logging
import os
import sys
import traceback
import warnings

TEST_PACKAGE = 'k3d.test'
SCENE_GLOB = 'test_visual_'


class NeedsBrowser(Exception):
    """Raised by a stub whose behaviour cannot be faked - the test is reported, not guessed at."""


class _Browser:
    def execute_script(self, *args, **kwargs):
        return None

    def get_log(self, *args, **kwargs):
        return []

    def save_screenshot(self, *args, **kwargs):
        raise NeedsBrowser('save_screenshot')


class _Headless:
    """Everything the visual tests ask of the harness, doing nothing."""

    def __init__(self):
        self.browser = _Browser()

    def sync(self, *args, **kwargs):
        return None

    def camera_reset(self, *args, **kwargs):
        return None

    def get_screenshot(self, *args, **kwargs):
        raise NeedsBrowser('get_screenshot')

    def __getattr__(self, name):
        raise NeedsBrowser(name)


def _install_stubs():
    import pytest

    import k3d

    pytest.plot = k3d.plot()
    pytest.headless = _Headless()

    return pytest


def _install_compare(scenes, compression_level):
    """Rebind plot_compare.compare so a comparison becomes a snapshot on disk."""
    import pytest

    plot_compare = importlib.import_module(TEST_PACKAGE + '.plot_compare')

    def compare(name, only_canvas=True, threshold=0.2, max_mismatched_pixels=0,
                camera_factor=1.0, modes=('simple', 'advanced', 'cinematic')):
        # a test may compare more than once under one name - a state before an update and after -
        # and those are different scenes. Numbering the repeats keeps both instead of overwriting.
        taken = sum(1 for scene in scenes if scene['base'] == name)

        scenes.append({
            'base': name,
            'name': name if taken == 0 else '%s__%d' % (name, taken + 1),
            'modes': list(modes),
            'blob': pytest.plot.get_binary_snapshot(compression_level=compression_level),
            'objects': len(pytest.plot.objects),
            # the k3d object types in the scene, so results can be grouped by what was drawn
            # rather than by what the test happened to be called
            'types': sorted({getattr(o, 'type', None) for o in pytest.plot.objects
                             if getattr(o, 'type', None)}),
        })

    plot_compare.compare = compare

    return plot_compare


def _scene_modules():
    test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return sorted(
        name[:-3] for name in os.listdir(test_dir)
        if name.startswith(SCENE_GLOB) and name.endswith('.py')
    )


def _cases(module):
    """Test functions with their parametrisations, in declaration order."""
    out = []

    for name in sorted(vars(module)):
        if not name.startswith('test_'):
            continue

        function = getattr(module, name)

        if not callable(function):
            continue

        params = []

        for mark in getattr(function, 'pytestmark', []):
            if mark.name == 'parametrize':
                names = [n.strip() for n in mark.args[0].split(',')]
                params = [
                    dict(zip(names, values if isinstance(values, (tuple, list)) else (values,)))
                    for values in mark.args[1]
                ]

        out.append((name, function, params or [{}]))

    return out


def generate(out_dir, compression_level=1, verbose=True):
    """Run every visual test against the stubs and write what it built.

    Returns (manifest, report) - the manifest is what the runner consumes, the report says which
    tests could not be replayed without a browser and why.
    """
    pytest = _install_stubs()
    scenes = []
    _install_compare(scenes, compression_level)

    # every k3d module sets its own logger level, so silencing the parent achieves nothing -
    # the assets a hundred and forty tests touch would print a hundred and forty times
    quieted = {}

    for name in list(logging.Logger.manager.loggerDict):
        if name == 'k3d' or name.startswith('k3d.'):
            logger = logging.getLogger(name)
            quieted[name] = logger.level
            logger.setLevel(logging.WARNING)

    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    report = []

    for module_name in _scene_modules():
        try:
            module = importlib.import_module('%s.%s' % (TEST_PACKAGE, module_name))
        except Exception as exc:  # noqa: BLE001
            # a test module can depend on something this interpreter cannot import - scikit-image
            # built against another numpy, say. Its scenes are missing and that is worth saying,
            # but it is no reason to abandon the other hundred and forty.
            report.append((module_name, '(import)', 'import failed',
                           '%s: %s' % (type(exc).__name__, str(exc)[:100])))
            continue

        for case_name, function, parametrisations in _cases(module):
            for kwargs in parametrisations:
                label = case_name + (('[' + ','.join('%s=%s' % kv for kv in kwargs.items()) + ']')
                                     if kwargs else '')
                before = len(scenes)

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        function(**kwargs)
                    status = 'ok'
                    detail = '%d scene(s)' % (len(scenes) - before)
                except NeedsBrowser as exc:
                    status = 'needs a browser'
                    detail = str(exc)
                except Exception as exc:  # noqa: BLE001 - a scene we cannot build is data, not a crash
                    status = 'error'
                    detail = '%s: %s' % (type(exc).__name__, str(exc)[:120])

                    if verbose and os.environ.get('K3D_CORPUS_TRACE'):
                        traceback.print_exc()

                report.append((module_name, label, status, detail))

    # the cost of a frame that draws nothing: JS, camera, compositing. Subtracted from every
    # scene, so what gets reported is the object's own work and not the harness's.
    import k3d

    empty = k3d.plot()
    scenes.append({
        'base': '__empty__',
        'name': '__empty__',
        'modes': ['simple'],
        'blob': empty.get_binary_snapshot(compression_level=compression_level),
        'objects': 0,
    })

    for scene in scenes:
        path = os.path.join(out_dir, scene['name'] + '.k3d')

        with open(path, 'wb') as handle:
            handle.write(scene['blob'])

        manifest.append({
            'name': scene['name'],
            'file': scene['name'] + '.k3d',
            'bytes': len(scene['blob']),
            'objects': scene['objects'],
            'types': scene.get('types', []),
            'modes': scene['modes'],
            'base': scene['base'],
        })

    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump({'scenes': manifest}, handle, indent=1)

    for name, level in quieted.items():
        logging.getLogger(name).setLevel(level)

    return manifest, report


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'scenes')
    manifest, report = generate(out_dir)

    counts = {}
    for _, _, status, _ in report:
        counts[status] = counts.get(status, 0) + 1

    print('%-34s %-42s %-24s %s' % ('module', 'test', 'status', 'detail'))
    for module_name, label, status, detail in report:
        if status != 'ok':
            print('%-34s %-42s %-24s %s' % (module_name, label[:42], status, detail))

    print()
    print('tests: %s' % ', '.join('%s=%d' % kv for kv in sorted(counts.items())))
    print('scenes written: %d, %.1f MB total'
          % (len(manifest), sum(s['bytes'] for s in manifest) / 1e6))
    print('directory: %s' % out_dir)


if __name__ == '__main__':
    main()
