'use strict';
var getSSAAChunkedRender = require('./../helpers/SSAAChunkedRender');

/**
 * @memberof K3D.Providers.ThreeJS.Initializers
 * @inner
 * @param  {K3D.Core} K3D       Current K3D instance
 * @param  {Object} on          Internal type of listener (when its called)
 * @param  {Function} listener  Listener to be removed
 */
function handleListeners(K3D, on, listener) {

    listener.call(K3D);

    if (listener.callOnce) {
        K3D.removeFrameUpdateListener(on, listener);
    }
}

/**
 * Renderer initializer for Three.js library
 * @this K3D.Core world
 * @method Renderer
 * @memberof K3D.Providers.ThreeJS.Initializers
 * @param {Object} K3D current K3D instance
 */
module.exports = function (K3D) {

    var self = this, renderingPromise = null,
        canvas = document.createElement('canvas'),
        context = canvas.getContext('webgl2', {
            antialias: K3D.parameters.antialias,
            preserveDrawingBuffer: false,
            alpha: true,
            powerPreference: 'high-performance'
        }),
        gl, debugInfo;

    self.renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        context: context
    });

    canvas.addEventListener('webglcontextlost', function (event) {
        event.preventDefault();
        console.log(event);
    }, false);

    gl = self.renderer.context;

    debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    console.log('K3D: (UNMASKED_VENDOR_WEBGL)', gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL));
    console.log('K3D: (UNMASKED_RENDERER_WEBGL)', gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL));

    function render() {
        return new Promise(function (resolve) {
            if (K3D.disabling) {
                return void(0);
            }

            var size = new THREE.Vector2(), chunk_widths = [];

            self.renderer.getSize(size);

            if (K3D.parameters.renderingSteps > 1) {
                var s = size.x / K3D.parameters.renderingSteps;

                for (var i = 0; i < K3D.parameters.renderingSteps; i++) {
                    var o1 = Math.round(i * s);
                    var o2 = Math.min(Math.round((i + 1) * s), size.x);
                    chunk_widths.push([o1, o2 - o1]);
                }
            }

            K3D.frameUpdateHandlers.before.forEach(handleListeners.bind(null, K3D, 'before'));

            K3D.refreshGrid();

            self.renderer.clippingPlanes = [];

            self.camera.updateMatrixWorld();
            self.renderer.clear();
            self.renderer.render(self.gridScene, self.camera);

            K3D.parameters.clippingPlanes.forEach(function (plane) {
                self.renderer.clippingPlanes.push(new THREE.Plane(new THREE.Vector3().fromArray(plane), plane[3]));
            });

            K3D.dispatch(K3D.events.BEFORE_RENDER);

            var p = Promise.resolve();

            if (K3D.parameters.renderingSteps > 1) {
                self.controls.enabled = false;

                chunk_widths.forEach(function (c) {
                    p = p.then(function () {
                        self.renderer.setViewport(c[0], 0, c[1], size.y);
                        self.camera.setViewOffset(size.x, size.y, c[0], 0, c[1], size.y);
                        self.renderer.render(self.scene, self.camera);
                    });

                    p = p.then(function () {
                        return new Promise(function (resolve) {
                            setTimeout(resolve, 50);
                        });
                    });
                });
            } else {
                p = p.then(function () {
                    self.renderer.render(self.scene, self.camera);
                });
            }

            p = p.then(function () {
                self.controls.enabled = true;

                self.renderer.setViewport(0, 0, size.x, size.y);
                self.camera.clearViewOffset();

                K3D.frameUpdateHandlers.after.forEach(handleListeners.bind(null, K3D, 'after'));

                K3D.dispatch(K3D.events.RENDERED);

                if (K3D.autoRendering) {
                    requestAnimationFrame(render);
                } else {
                    resolve(true);
                }
            });
        });
    }

    this.renderer.setClearColor(0, 0);
    this.renderer.autoClear = false;

    this.render = function (force) {
        if (!K3D.autoRendering || force) {
            if (renderingPromise === null) {
                renderingPromise = render().then(function () {
                    renderingPromise = null;
                });

                return renderingPromise;
            } else if (force) {
                renderingPromise = renderingPromise.then(render).then(function () {
                    renderingPromise = null;
                });
            }
        }
    };

    this.renderOffScreen = function (width, height) {
        var rt,
            chunk_heights = [],
            chunk_count = Math.max(Math.min(128, K3D.parameters.renderingSteps), 1);

        var s = height / chunk_count;

        for (var i = 0; i < chunk_count; i++) {
            var o1 = Math.round(i * s);
            var o2 = Math.min(Math.round((i + 1) * s), height);
            chunk_heights.push([o1, o2 - o1]);
        }

        rt = new THREE.WebGLRenderTarget(width, Math.ceil(height / chunk_count));
        self.renderer.clippingPlanes = [];

        return getSSAAChunkedRender(self.renderer, self.gridScene, self.camera,
            rt, width, height, chunk_heights, 5).then(function (grid) {

                K3D.parameters.clippingPlanes.forEach(function (plane) {
                    self.renderer.clippingPlanes.push(new THREE.Plane(new THREE.Vector3().fromArray(plane), plane[3]));
                });

                return getSSAAChunkedRender(self.renderer, self.scene, self.camera,
                    rt, width, height, chunk_heights, 5).then(function (scene) {
                        rt.dispose();
                        return [grid, scene];
                    }
                );
            }
        );
    };
};
