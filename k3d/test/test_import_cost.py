import os
import subprocess
import sys

import k3d


def _modules_after_import_k3d(prefix):
    """How many modules with this prefix `import k3d` leaves behind, in a fresh interpreter."""
    probe = (
        "import k3d, sys; "
        "print(len([m for m in sys.modules if m.split('.')[0] == %r]))" % prefix
    )
    # the suite runs from inside the package directory and imports k3d off sys.path (conftest),
    # so the subprocess is given the same root rather than relying on an installed copy
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(k3d.__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")

    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)

    assert out.returncode == 0, out.stderr

    return int(out.stdout.strip())


def test_importing_k3d_does_not_import_vtk():
    # vtk is an optional dependency of one factory out of 23 and roughly a third of the time
    # `import k3d` takes for anyone who has it - which includes everyone who installed pyvista
    # for something else. A subprocess, because this session has already imported both.
    assert _modules_after_import_k3d("vtk") == 0
