import atexit
import copy
import logging
import threading
import time
from base64 import b64decode

import msgpack
import numpy as np
from deepcomparer import deep_compare
from flask import Flask, send_from_directory
from werkzeug import Response
from werkzeug.serving import make_server

from .helpers import to_json

# Nothing here narrates its own success. A notebook calls sync() once per frame of an animation,
# and a breadcrumb per call buries whatever the cell was asked to show; the same argument that
# silenced werkzeug's request log applies to our own. Warnings and errors stay at their level
# and are never suppressed. Raise this logger to DEBUG to get all of it back:
#
#     logging.getLogger('k3d.headless').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# logging.basicConfig(filename='test.log', level=logging.DEBUG)
class _QuietRequestHandler(WSGIRequestHandler):
    """Serves without narrating.

    The page polls /ping every few seconds for as long as the session is open - it is how it
    notices that the Python side went away - and werkzeug logs every request it answers. In a
    notebook that buries whatever the cell was actually asked to show. Raise this module's
    logger to DEBUG to get the request log back; errors are never suppressed, they go through
    log_error.
    """

    def log_request(self, code="-", size="-"):
        if logger.isEnabledFor(logging.DEBUG):
            super().log_request(code, size)



def _property_changed(current, synced, object_id, name):
    """Whether a synced property was edited.

    Arrays compare elementwise, everything else through deep_compare. Reaching for .any()
    first and catching the failure logged two warnings per scalar property per sync.
    """
    if isinstance(current, np.ndarray) and isinstance(synced, np.ndarray):
        if current.shape != synced.shape:
            return True

        try:
            return bool((current != synced).any())
        except Exception as e:
            logger.warning(
                f"Array comparison failed for object {object_id} property {name}: {e}"
            )

            return True

    if isinstance(current, np.ndarray) or isinstance(synced, np.ndarray):
        return True

    try:
        return not deep_compare(current, synced)
    except Exception as e:
        logger.warning(
            f"Comparison failed for object {object_id} property {name}: {e}"
        )

        return True


DEFAULT_STARTUP_TIMEOUT = 60.0
DEFAULT_REFRESH_TIMEOUT = 120.0


def _browser_errors(driver):
    """Best-effort JS console errors, to explain a timeout instead of just reporting one."""
    try:
        entries = driver.get_log("browser")
    except Exception:
        return ""  # not supported by every driver (e.g. Firefox)

    messages = [
        e.get("message", "") for e in entries if e.get("level") in ("SEVERE", "ERROR")
    ]
    if not messages:
        return ""

    return " Browser console errors: " + " | ".join(messages[-5:])


class k3d_remote:
    def __init__(
            self,
            k3d_plot,
            driver,
            width=1280,
            height=720,
            port=8080,
            startup_timeout=DEFAULT_STARTUP_TIMEOUT,
            refresh_timeout=DEFAULT_REFRESH_TIMEOUT,
    ):

        driver.set_window_size(width, height)

        self.port = port
        self.browser = driver
        self.k3d_plot = k3d_plot
        self.refresh_timeout = refresh_timeout

        self.api = Flask(__name__)

        self.server = make_server(
            "localhost", port, self.api, request_handler=_QuietRequestHandler
        )

        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(), daemon=True
        )
        self.thread.daemon = True
        self.thread.start()

        self.synced_plot = dict.fromkeys(k3d_plot.get_plot_params().keys())
        self.synced_objects = {}

        @self.api.route("/<path:path>")
        def static_file(path):
            root_dir = self.k3d_plot.get_static_path()
            return send_from_directory(root_dir, path)

        @self.api.route("/ping")
        def ping():
            return Response(":)")

        @self.api.route("/", methods=["POST"])
        def generate():
            try:
                diff = self._sync.diff()
                logger.debug("Generated plot diff and objects diff for sync.")
                return Response(
                    msgpack.packb(diff, use_bin_type=True),
                    mimetype="application/octet-stream",
                )
            except Exception as e:
                logger.error(f"Error in generate route: {e}")
                raise

        deadline = time.monotonic() + startup_timeout
        while not self.browser.execute_script(
                "return typeof(window.headlessK3D) !== 'undefined'"
        ):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "window.headlessK3D was not defined within %g s - headless.html "
                    "failed to load or the bundle raised.%s"
                    % (startup_timeout, _browser_errors(self.browser))
                )
            time.sleep(1)
            self.browser.get(url="http://localhost:" + str(port) + "/headless.html")

        self.browser.execute_script(f"window.init({width}, {height});")

        atexit.register(self.close)

    def sync(self, hold_until_refreshed=False):
        # Check if k3dRefresh exists and run only then. Probe up to 5 times before exception.
        for _ in range(5):
            if self.browser.execute_script("return typeof(k3dRefresh) !== 'undefined'"):
                self.browser.execute_script("k3dRefresh()")
                logger.debug("k3dRefresh executed in browser.")
                break
            time.sleep(0.2)
        else:
            logger.error("k3dRefresh is not defined in the browser after 5 attempts.")
            raise RuntimeError(
                "k3dRefresh is not defined in the browser after 5 attempts."
            )

        if hold_until_refreshed:
            deadline = time.monotonic() + self.refresh_timeout
            while not self.browser.execute_script("return window.refreshed"):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "window.refreshed was not set within %g s - the render never "
                        "completed.%s"
                        % (self.refresh_timeout, _browser_errors(self.browser))
                    )
                time.sleep(0.1)

    def get_browser_screenshot(self):
        return self.browser.get_screenshot_as_png()

    def camera_reset(self, factor=1.5):
        self.browser.execute_script("K3DInstance.resetCamera(%f)" % factor)
        # refresh dom elements
        self.browser.execute_script("K3DInstance.refreshGrid()")
        self.browser.execute_script("K3DInstance.dispatch(K3DInstance.events.RENDERED)")

    def get_screenshot(self, only_canvas=False):
        screenshot = self.browser.execute_script(
            """
        return K3DInstance.getScreenshot(K3DInstance.parameters.screenshotScale, %d).then(function (d){
        return d.toDataURL().split(',')[1];
        });
        """
            % only_canvas
        )

        if status != 200:
            raise RuntimeError("the page could not post its screenshot (HTTP %s)" % status)

        # the body lands on the server thread, so it can arrive after the script resolves
        if not self._screenshot_ready.wait(timeout or self.refresh_timeout):
            raise TimeoutError(
                "the screenshot was rendered but never arrived over the HTTP channel"
            )

        return self._screenshot

    def get_gltf(self):
        """Return the scene geometry as a binary glTF (.glb).

        See Plot.fetch_gltf for what a glTF can and cannot carry over from a K3D scene.
        """
        gltf = self.browser.execute_script(
            """
        return K3DInstance.getGLTF().then(function (glb) {
            var bytes = new Uint8Array(glb);
            var chunks = [];

            // one apply() over the whole buffer overflows the argument stack on real meshes
            for (var i = 0; i < bytes.length; i += 8192) {
                chunks.push(String.fromCharCode.apply(null, bytes.subarray(i, i + 8192)));
            }

            return btoa(chunks.join(''));
        });
        """
        )

        return b64decode(gltf)

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server = None

        if self.browser is not None:
            # quit(), not close(): close() only closes the current window and would leave the
            # WebDriver session and the chromedriver/browser processes behind.
            self.browser.quit()
            self.browser = None


def _relax_timeouts(driver):
    """Cinematic screenshots block inside a single execute_script call for tens of minutes.
    Both limits are raised: the 120 s HTTP timeout is the effective bound, since a promise
    returned from execute_script is not an async script and the script timeout never fires."""
    driver.set_script_timeout(3600)

    client_config = getattr(driver.command_executor, "_client_config", None)

    if client_config is not None:
        client_config.timeout = 3600

    return driver


def get_headless_driver(no_headless=False, gpu=False):
    from selenium import webdriver

    options = webdriver.ChromeOptions()

    options.add_argument("--no-sandbox")

    if not no_headless:
        if gpu:
            options.add_argument("--headless=new")
            options.add_argument("--ignore-gpu-blocklist")
            options.add_argument("--enable-webgl")
        else:
            options.add_argument("--headless")
            options.add_argument("--enable-unsafe-swiftshader")

    return _relax_timeouts(webdriver.Chrome(options=options))


def get_headless_firefox_driver(no_headless=False):
    from selenium import webdriver

    options = webdriver.FirefoxOptions()

    options.add_argument("--no-sandbox")

    if not no_headless:
        options.add_argument("--headless")
        options.add_argument("--enable-unsafe-swiftshader")

    return _relax_timeouts(webdriver.Firefox(options=options))
    def get_memory(self, collect=True):
        """The page's JS heap in megabytes, or None where the browser does not report it.

        `precise` and `collected` say whether the numbers can be trusted: without
        --enable-precise-memory-info Chrome answers with a frozen constant, and without
        --js-flags=--expose-gc the reading carries the last frame's garbage. Pass both through
        get_headless_driver(extra_args=...). used against total is the fragmentation signal.
        """
        result = self.browser.execute_script(
            """
            var collect = arguments[0];
            var gc = typeof window.gc === 'function';

            if (collect && gc) { window.gc(); window.gc(); }

            if (!performance.memory) { return null; }

            return [performance.memory.usedJSHeapSize,
                    performance.memory.totalJSHeapSize,
                    performance.memory.jsHeapSizeLimit,
                    gc];
            """,
            bool(collect),
        )

        if result is None:
            return None

        used, total, limit, has_gc = result
        mb = 1024.0 * 1024.0

        return {
            "used_mb": used / mb,
            "total_mb": total / mb,
            "limit_mb": limit / mb,
            "collected": bool(collect and has_gc),
            "precise": self._memory_is_precise(),
        }

    def _memory_is_precise(self, probe_mb=24):
        """Whether usedJSHeapSize follows what the page allocates.

        Tested, not guessed: the quantised answer is a fixed 10000000 bytes, which no property of
        the number reveals. The probe chunk is released before returning.
        """
        if self._precise_memory is None:
            self._precise_memory = bool(self.browser.execute_script(
                """
                var mb = arguments[0];

                if (!performance.memory) { return false; }

                var before = performance.memory.usedJSHeapSize;
                var probe = new Uint8Array(mb * 1048576);

                // touched, or the pages are never committed and the reading would not move even
                // where the counter is honest
                for (var i = 0; i < probe.length; i += 4096) { probe[i] = 1; }

                var after = performance.memory.usedJSHeapSize;

                probe = null;

                return (after - before) > (mb * 1048576) * 0.6;
                """,
                probe_mb,
            ))

        return self._precise_memory

    def get_gl_info(self):
        """What the browser is actually rendering with.

        A container that loses its GPU passthrough falls back to software rendering without
        failing, and every timing taken there is meaningless while every image still looks
        right. Returns None when the page has no plot yet.
        """
        return self.browser.execute_script(
            "return typeof K3DInstance !== 'undefined' && K3DInstance "
            "? K3DInstance.glInfo : null;"
        )

