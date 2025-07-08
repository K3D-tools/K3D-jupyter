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


def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """
    # process = subprocess.Popen("webpack", cwd=os.path.abspath("./../js/"), shell=True)
    # process.wait()


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    pytest.plot = k3d.plot(
        screenshot_scale=1.0, antialias=2, camera_auto_fit=False, colorbar_object_id=0
    )
    print(pytest.plot.get_static_path())
    pytest.headless = k3d_remote(pytest.plot, get_headless_driver())
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
