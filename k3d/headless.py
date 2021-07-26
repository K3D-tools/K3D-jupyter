import threading
import numpy as np
import copy
import msgpack
from flask import Flask, request, Response, send_from_directory
from base64 import b64decode
import logging
import atexit
import urllib.request

from .helpers import to_json

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class k3d_remote:
    def __init__(self, k3d_plot, driver, width=1280, height=720, port=8080):

        driver.set_window_size(width, height)

        self.port = port
        self.browser = driver
        self.k3d_plot = k3d_plot

        self.api = Flask(__name__)

        thread = threading.Thread(target=lambda: self.api.run(host="0.0.0.0", port=port),
                                  daemon=True)
        thread.deamon = True
        thread.start()

        self.synced_plot = {k: None for k in k3d_plot.get_snapshot_params().keys()}
        self.synced_objects = {}

        @self.api.route('/stop', methods=['GET'])
        def stopServer():
            shutdown_hook = request.environ.get('werkzeug.server.shutdown')

            if shutdown_hook is not None:
                shutdown_hook()

            return Response("Bye", mimetype='text/plain')

        @self.api.route('/<path:path>')
        def static_file(path):
            root_dir = self.k3d_plot.get_static_path()
            return send_from_directory(root_dir, path)

        @self.api.route('/ping')
        def ping():
            return Response(":)")

        @self.api.route('/', methods=['POST'])
        def generate():
            current_plot_params = self.k3d_plot.get_snapshot_params()
            plot_diff = {k: current_plot_params[k] for k in current_plot_params.keys()
                         if current_plot_params[k] != self.synced_plot[k]}

            objects_diff = {}

            for o in self.k3d_plot.objects:
                if o.id not in self.synced_objects:
                    objects_diff[o.id] = {k: to_json(k, o[k]) for k in o.keys if
                                          not k.startswith('_')}
                else:
                    for p in o.keys:
                        if p.startswith('_'):
                            continue

                        sync = False

                        if isinstance(o[p], np.ndarray):
                            sync = (o[p] != self.synced_objects[o.id][p]).any()
                        else:
                            sync = o[p] != self.synced_objects[o.id][p]

                        if sync:
                            if o.id not in objects_diff.keys():
                                objects_diff[o.id] = {"id": o.id, "type": o.type}

                            objects_diff[o.id][p] = to_json(p, o[p])

            diff = {
                "plot_diff": plot_diff,
                "objects_diff": objects_diff
            }

            self.synced_objects = {v.id: {k: copy.deepcopy(v[k]) for k in v.keys} for v in
                                   self.k3d_plot.objects}
            self.synced_plot = current_plot_params

            return (msgpack.packb(diff, use_bin_type=True), 200)

        self.browser.implicitly_wait(5)
        self.browser.get(url="http://localhost:" + str(port) + "/headless.html")

        atexit.register(self.close)

    def sync(self):
        self.browser.execute_script("k3dRefresh()")

    def get_browser_screenshot(self):
        return self.browser.get_screenshot_as_png()

    def get_screenshot(self, only_canvas=False):
        screenshot = self.browser.execute_script("""
        return K3DInstance.getScreenshot(K3DInstance.parameters.screenshotScale, %d).then(function (d){
        return d.toDataURL().split(',')[1];
        });                                 
        """ % only_canvas)

        return b64decode(screenshot)

    def close(self):
        urllib.request.urlopen("http://localhost:" + str(self.port) + "/stop")
        self.browser.close()


def get_headless_driver(no_headless = False):
    from selenium.webdriver.chrome.options import Options
    from selenium import webdriver
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()

    if not no_headless:
        options.add_argument("--headless")

    return webdriver.Chrome(ChromeDriverManager().install(), options=options)
