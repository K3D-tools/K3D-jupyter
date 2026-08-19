import inspect
import os
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import subprocess

import pytest

import k3d
from k3d.headless import get_headless_driver, k3d_remote


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="run tests with GPU support")


def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """
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


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    pytest.plot = k3d.plot(
        screenshot_scale=1.0, antialias=2, camera_auto_fit=False, colorbar_object_id=0
    )
    print(pytest.plot.get_static_path())
    gpu = session.config.getoption("--gpu")
    driver = get_headless_driver(gpu=gpu)

    # Two independent clocks govern a screenshot: how long the browser may spend
    # running the script, and how long the client waits for the HTTP response
    # carrying its result. A cinematic reference accumulates its whole sample
    # budget inside one call, so both have to allow minutes - raising only the
    # first leaves the second (120s by default) to abort heavy scenes mid-render.
    driver.set_script_timeout(600)

    client_config = getattr(driver.command_executor, "_client_config", None)
    if client_config is not None:
        client_config.timeout = 900
    pytest.headless = k3d_remote(pytest.plot, driver)
    pytest.headless.browser.execute_script("window.randomMul = 0.0;")


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """

    pytest.headless.close()


def pytest_unconfigure(config):
    """
    called before test process is exited.
    """
