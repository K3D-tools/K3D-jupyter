"""Put a given k3d version's browser bundle where the runner page can load it.

Two files per version are enough - require.js and standalone.js - because the runner does what a
standalone snapshot does: define the AMD module and ask for it. Nothing here looks inside the
bundle, which is the point: it is minified and every version minifies differently.

`local` means the working tree's own build, k3d/static. Anything else is a released version and is
taken out of its wheel, so measuring against an old release needs no environment of its own.

    python -m k3d.test.browser_performance.bundles local 2.18.0
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLES_DIR = os.path.join(HERE, 'bundles')
FILES = ('require.js', 'standalone.js')


def _repo_static():
    return os.path.abspath(os.path.join(HERE, '..', '..', 'static'))


def _from_directory(source, target):
    os.makedirs(target, exist_ok=True)

    for name in FILES:
        origin = os.path.join(source, name)

        if not os.path.isfile(origin):
            raise FileNotFoundError('%s not in %s' % (name, source))

        shutil.copyfile(origin, os.path.join(target, name))


def _from_wheel(version, target):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'download', '--no-deps', '--only-binary', ':all:',
             '-d', tmp, 'k3d==%s' % version],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )

        wheels = glob.glob(os.path.join(tmp, '*.whl'))

        if not wheels:
            raise RuntimeError('pip download brought no wheel for k3d==%s' % version)

        os.makedirs(target, exist_ok=True)

        with zipfile.ZipFile(wheels[0]) as archive:
            for name in FILES:
                member = 'k3d/static/' + name

                if member not in archive.namelist():
                    raise FileNotFoundError('%s not in %s' % (member, os.path.basename(wheels[0])))

                with archive.open(member) as source, \
                        open(os.path.join(target, name), 'wb') as handle:
                    shutil.copyfileobj(source, handle)


def ensure(version, force=False):
    """Return the directory holding `version`'s bundle, fetching it if need be."""
    target = os.path.join(BUNDLES_DIR, version)
    complete = all(os.path.isfile(os.path.join(target, name)) for name in FILES)

    # a release is immutable and worth caching, the working tree's build is not - keeping a copy
    # of it would measure whatever was built last time instead of what is checked out now
    if complete and not force and version != 'local':
        return target

    if version == 'local':
        _from_directory(_repo_static(), target)
    else:
        _from_wheel(version, target)

    return target


def order(names):
    """Oldest release first, `local` last.

    The order is the measurement order, and the first entry is the one that decides how many copies
    a scene is measured with - so it has to be the oldest, the one most likely to need the most.
    Alphabetical would put 10.0.0 before 2.18.0, hence the numeric key.
    """
    def key(name):
        if name == 'local':
            return (1, ())

        parts = []

        for chunk in name.replace('-', '.').split('.'):
            parts.append((0, int(chunk)) if chunk.isdigit() else (1, 0, chunk))

        return (0, tuple(parts))

    return sorted(names, key=key)


def available():
    if not os.path.isdir(BUNDLES_DIR):
        return []

    return order(
        name for name in os.listdir(BUNDLES_DIR)
        if all(os.path.isfile(os.path.join(BUNDLES_DIR, name, f)) for f in FILES)
    )


def main():
    versions = sys.argv[1:] or ['local']

    for version in versions:
        target = ensure(version)
        sizes = ', '.join('%s %.1f kB' % (name, os.path.getsize(os.path.join(target, name)) / 1e3)
                          for name in FILES)
        print('%-10s %s   (%s)' % (version, target, sizes))


if __name__ == '__main__':
    main()
