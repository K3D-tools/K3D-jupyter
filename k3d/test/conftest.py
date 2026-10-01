import inspect
import os
import shutil
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import subprocess

import pytest

import k3d
from k3d.headless import get_headless_driver, k3d_remote

from .plot_compare import capture_baseline, check_environment


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="run tests with GPU support")
    parser.addoption(
        "--no-browser", action="store_true", default=False,
        help="skip every test that reaches for the browser, instead of starting one",
    )


def _accepting():
    """Modes whose renders this run will overwrite instead of asserting."""
    return [m.strip() for m in os.environ.get("K3D_ACCEPT_REFERENCES", "").split(",") if m.strip()]


# The bundle and the browser are built on first use, not at session start. 123 of the 338 tests
# never open a browser and 103 never read the bundle either, so building both up front is what
# made checking a traitlets change need Chrome, node and a 17-second webpack run.
_BUILT = {"bundle": False, "harness": False, "gpu": False}


def build_bundle():
    """Run webpack, once per session. A test that reads k3d/static asks for the bundle fixture."""
    if _BUILT["bundle"]:
        return

    _BUILT["bundle"] = True

    # An xdist worker never builds: the controller already did, in pytest_configure_node. Two
    # webpacks writing k3d/static at once leave a bundle that loads and then throws, which reads
    # as a dozen unrelated visual failures rather than as a build problem.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    # Only run webpack if the directory exists (e.g. not in installed package)
    js_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../js"))
    if os.path.exists(js_dir) and os.path.isdir(js_dir):
        # Check if webpack is installed/available before trying to run it
        try:
            # use npm run build which is cross-platform and uses project's webpack
            if sys.platform == "win32":
                process = subprocess.Popen("npm run build", cwd=js_dir, shell=True)
            else:
                process = subprocess.Popen(["npm", "run", "build"], cwd=js_dir)
            returncode = process.wait()
        except FileNotFoundError:
            print("Skipping webpack build (npm not found or js dir missing)")
        else:
            if returncode != 0:
                pytest.exit(
                    "webpack build failed (npm run build exited %d) - refusing to run "
                    "the suite against a stale JS bundle" % returncode,
                    returncode=1,
                )
    else:
        print(f"Skipping webpack build: {js_dir} not found")


def _build_harness():
    """Bring up the browser and the plot every visual test shares."""
    if _BUILT["harness"]:
        return

    _BUILT["harness"] = True
    build_bundle()

    plot = k3d.plot(
        screenshot_scale=1.0, antialias=2, camera_auto_fit=False, colorbar_object_id=0,
        cinematic_seed=1
    )
    print(plot.get_static_path())

    # get_headless_driver already raises both timeouts to an hour, which is what a cinematic
    # screenshot needs: it blocks inside one execute_script call, and the HTTP timeout is the
    # bound that actually fires. Lowering them here undid exactly that.
    driver = get_headless_driver(gpu=_BUILT["gpu"])

    # port 0: xdist workers are separate processes and would otherwise fight over 8080
    remote = k3d_remote(plot, driver, port=0)
    remote.browser.execute_script("window.randomMul = 0.0;")

    # every test starts from this state, and prepare() restores all of it
    capture_baseline(plot)

    # one sync so the page has a plot to report its renderer from
    remote.sync(hold_until_refreshed=True)
    check_environment(remote)

    # published together and last: a half-built harness must not read as a built one
    pytest.plot = plot
    pytest.headless = remote


# optionalhook: xdist defines this hookspec, and without xdist installed pytest would reject
# the whole conftest rather than ignore a hook it does not know.
@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    """xdist controller: build the bundle before any worker starts. Workers never build.

    -n is faster - 1377 s -> 671 s at -n 4, -> 591 s at -n 8 - but it is not equivalent to a
    serial run. Eleven tests fail at -n 4 and thirteen at -n 8, and mip, text, text2d and
    vector_field fail at both, in the simple renderer and by thousands of pixels, while each of
    those files passes alone and passes in the full serial order. Some file leaves the browser in
    a state they depend on, and which files share a browser is what -n changes. Until that is
    found, -n is for a quick local sweep and a red test under it is re-checked serially before it
    is believed; CI stays serial.
    """
    build_bundle()


@pytest.fixture(scope="session")
def bundle():
    """For a test that reads what webpack emitted without opening a browser."""
    build_bundle()


def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """
    # pytest.plot and pytest.headless are read directly by 567 lines across 60 files. Rather than
    # rewrite them as fixtures, the first read builds the harness: PEP 562 lookup only fires for
    # names the module does not have, and _build_harness sets both, so it fires exactly once.
    inherited = pytest.__dict__.get("__getattr__")

    def __getattr__(name):
        if name in ("plot", "headless"):
            # The set of tests that need a browser is discovered by running, not maintained as a
            # list: reaching for the harness is what needing one means.
            if config.getoption("--no-browser"):
                pytest.skip("needs the browser harness")

            _build_harness()

            if name not in pytest.__dict__:
                raise RuntimeError("the test harness failed to start; see the error above")

            return pytest.__dict__[name]

        if inherited is not None:
            return inherited(name)

        raise AttributeError("module 'pytest' has no attribute %r" % name)

    pytest.__getattr__ = __getattr__

    # Accepting references asserts nothing. Doing it on a machine whose renderer nobody
    # recorded is how a reference drifts; doing it in CI would rewrite the ground truth from
    # whatever browser the runner happened to ship.
    if _accepting() and (os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")):
        pytest.exit(
            "K3D_ACCEPT_REFERENCES is set in CI. References are written by hand, in the pinned "
            "image, and reviewed as a diff - never by an automated run.",
            returncode=2,
        )

    # Writing references is a deliberate, reviewed act. Spread over workers the report of what
    # was overwritten arrives in pieces, out of order, from processes whose exit code the
    # controller replaces - so the one safeguard against a silent rewrite stops working.
    if _accepting() and getattr(config.option, "numprocesses", None):
        pytest.exit(
            "K3D_ACCEPT_REFERENCES cannot be combined with -n: an accepting run has to say "
            "exactly what it overwrote, and that report does not survive being split across "
            "workers. Run it serially.",
            returncode=2,
        )


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    # only this run's failures belong here: images from a previous one read as current evidence.
    # Under xdist this hook runs in every worker too, and a worker wiping the directory would
    # take the other workers' evidence with it - the controller runs first, so it does it once.
    if not hasattr(session.config, "workerinput"):
        results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

        if os.path.isdir(results):
            shutil.rmtree(results, ignore_errors=True)

    _BUILT["gpu"] = session.config.getoption("--gpu")


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    """Collect what a worker found: its ENVIRONMENT_MISMATCH lives in its own process."""
    from .plot_compare import ENVIRONMENT_MISMATCH

    for entry in (getattr(node, "workeroutput", None) or {}).get("k3d_env_mismatch") or []:
        if tuple(entry) not in {tuple(e) for e in ENVIRONMENT_MISMATCH}:
            ENVIRONMENT_MISMATCH.append(tuple(entry))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Say plainly how much of this run asserted nothing, and what drew it."""
    from .plot_compare import ACCEPTED, ENVIRONMENT_MISMATCH

    if ENVIRONMENT_MISMATCH:
        terminalreporter.write_sep("=", "RENDERER DOES NOT MATCH THE REFERENCES", red=True, bold=True)
        for key, want, got in ENVIRONMENT_MISMATCH:
            terminalreporter.write_line("  %s: references were drawn with %r, this run has %r"
                                        % (key, want, got))
        terminalreporter.write_line(
            "Every visual comparison in this run comes from a different rasterizer than the "
            "committed references. Run the suite through docker compose, which pins both."
        )

    if ACCEPTED:
        terminalreporter.write_sep("=", "REFERENCES ACCEPTED", red=True, bold=True)
        terminalreporter.write_line(
            "%d reference image(s) overwritten; nothing was asserted for them. "
            "Review the diff before committing." % len(ACCEPTED)
        )
        for name in sorted(ACCEPTED):
            terminalreporter.write_line("  %s" % name)


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    from .plot_compare import ACCEPTED, ENVIRONMENT_MISMATCH

    # A worker's findings reach the controller only through this channel; it reports them.
    if hasattr(session.config, "workeroutput"):
        session.config.workeroutput["k3d_env_mismatch"] = ENVIRONMENT_MISMATCH

    # In CI a wrong rasterizer is not a warning: every image it compared was meaningless.
    if ENVIRONMENT_MISMATCH and (os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")):
        session.exitstatus = 4

    # A run that rewrote its own ground truth is not a passing run, whatever pytest thinks.
    if ACCEPTED and exitstatus == 0:
        session.exitstatus = 3

    if _BUILT["harness"]:
        pytest.headless.close()


def pytest_unconfigure(config):
    """
    called before test process is exited.
    """
