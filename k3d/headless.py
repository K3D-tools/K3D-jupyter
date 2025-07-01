import atexit
import copy
import logging
import msgpack
import threading
import time
from base64 import b64decode
from deepcomparer import deep_compare
from flask import Flask, send_from_directory
from werkzeug import Response
from werkzeug.serving import make_server

from .helpers import to_json

# Set up module-level logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# logging.basicConfig(filename='test.log', level=logging.DEBUG)

class k3d_remote:
    def __init__(self, k3d_plot, driver, width=1280, height=720, port=8080):

        driver.set_window_size(width, height)

        self.port = port
        self.browser = driver
        self.k3d_plot = k3d_plot

        self.api = Flask(__name__)

        self.server = make_server("localhost", port, self.api)

        self.thread = threading.Thread(target=lambda: self.server.serve_forever(), daemon=True)
        self.thread.daemon = True
        self.thread.start()

        self.synced_plot = {k: None for k in k3d_plot.get_plot_params().keys()}
        self.synced_objects = {}

        @self.api.route('/<path:path>')
        def static_file(path):
            root_dir = self.k3d_plot.get_static_path()
            return send_from_directory(root_dir, path)

        @self.api.route('/ping')
        def ping():
            return Response(":)")

        @self.api.route('/', methods=['POST'])
        def generate():
            try:
                current_plot_params = self.k3d_plot.get_plot_params()
                plot_diff = {k: current_plot_params[k] for k in current_plot_params.keys()
                             if current_plot_params[k] != self.synced_plot[k] and k != 'minimumFps'}
                objects_diff = {}
                for o in self.k3d_plot.objects:
                    if o.id not in self.synced_objects:
                        objects_diff[o.id] = {k: to_json(k, o[k], o) for k in o.keys if
                                              not k.startswith('_')}
                    else:
                        for p in o.keys:
                            if p.startswith('_'):
                                continue
                            if p == 'voxels_group':
                                sync = True
                            else:
                                try:
                                    sync = (o[p] != self.synced_objects[o.id][p]).any()
                                except Exception as e:
                                    logger.warning(f"Comparison failed for object {o.id} property {p}: {e}")
                                    try:
                                        sync = o[p].shape != self.synced_objects[o.id][p].shape
                                    except Exception as e2:
                                        logger.warning(f"Shape comparison failed for object {o.id} property {p}: {e2}")
                                        sync = not deep_compare(o[p], self.synced_objects[o.id][p])
                            if sync:
                                if o.id not in objects_diff.keys():
                                    objects_diff[o.id] = {"id": o.id, "type": o.type}
                                objects_diff[o.id][p] = to_json(p, o[p], o)
                for k in self.synced_objects.keys():
                    if k not in self.k3d_plot.object_ids:
                        objects_diff[k] = None  # to remove from plot
                diff = {
                    "plot_diff": plot_diff,
                    "objects_diff": objects_diff
                }
                self.synced_objects = {v.id: {k: copy.deepcopy(v[k]) for k in v.keys} for v in
                                       self.k3d_plot.objects}
                self.synced_plot = current_plot_params
                logger.info("Generated plot diff and objects diff for sync.")
                return Response(msgpack.packb(diff, use_bin_type=True),
                                mimetype='application/octet-stream')
            except Exception as e:
                logger.error(f"Error in generate route: {e}")
                raise

        while self.browser.execute_script(
                "return typeof(window.headlessK3D) !== 'undefined'") == False:
            time.sleep(1)
            self.browser.get(url="http://localhost:" + str(port) + "/headless.html")

        self.browser.execute_script(f"window.init({width}, {height});")

        atexit.register(self.close)

    def sync(self, hold_until_refreshed=False):
        # Check if k3dRefresh exists and run only then. Probe up to 5 times before exception.
        for _ in range(5):
            if self.browser.execute_script("return typeof(k3dRefresh) !== 'undefined'"):
                self.browser.execute_script("k3dRefresh()")
                logger.info("k3dRefresh executed in browser.")
                break
            time.sleep(0.2)
        else:
            logger.error("k3dRefresh is not defined in the browser after 5 attempts.")
            raise RuntimeError("k3dRefresh is not defined in the browser after 5 attempts.")

        if hold_until_refreshed:
            while self.browser.execute_script("return window.refreshed") == False:
                time.sleep(0.1)

    def get_browser_screenshot(self):
        return self.browser.get_screenshot_as_png()

    def camera_reset(self, factor=1.5):
        self.browser.execute_script("K3DInstance.resetCamera(%f)" % factor)
        # refresh dom elements
        self.browser.execute_script("K3DInstance.refreshGrid()")
        self.browser.execute_script("K3DInstance.dispatch(K3DInstance.events.RENDERED)")

    def get_screenshot(self, only_canvas=False):
        screenshot = self.browser.execute_script("""
        return K3DInstance.getScreenshot(K3DInstance.parameters.screenshotScale, %d).then(function (d){
        return d.toDataURL().split(',')[1];
        });                                 
        """ % only_canvas)

        return b64decode(screenshot)

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server = None

        if self.browser is not None:
            self.browser.close()
            self.browser = None


def get_headless_driver(no_headless=False):
    from selenium import webdriver

    options = webdriver.ChromeOptions()

    options.add_argument("--no-sandbox")

    if not no_headless:
        options.add_argument("--headless")
        options.add_argument("--enable-unsafe-swiftshader")

    return webdriver.Chrome(options=options)


def get_headless_firefox_driver(no_headless=False):
    from selenium import webdriver

    options = webdriver.FirefoxOptions()

    options.add_argument("--no-sandbox")

    if not no_headless:
        options.add_argument("--headless")
        options.add_argument("--enable-unsafe-swiftshader")

    return webdriver.Firefox(options=options)
