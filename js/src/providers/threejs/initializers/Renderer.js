const THREE = require('three');
const { error } = require('../../../core/lib/Error');
const getSSAAChunkedRender = require('../helpers/SSAAChunkedRender');

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
    const self = this;
    let renderingPromise = null;
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('webgl2', {
        antialias: K3D.parameters.antialias > 0,
        preserveDrawingBuffer: true,
        alpha: true,
        stencil: true,
        powerPreference: 'high-performance',
    });

    self.renderer = new THREE.WebGLRenderer({
        alpha: true,
        precision: "highp",
        premultipliedAlpha: true,
        antialias: K3D.parameters.antialias > 0,
        logarithmicDepthBuffer: K3D.parameters.logarithmicDepthBuffer,
        canvas,
        context,
    });

    if (!context) {
        if (typeof WebGL2RenderingContext !== 'undefined') {
            error(
                'Your browser appears to support WebGL2 but it might ' +
                'be disabled. Try updating your OS and/or video card driver.',
                true);
        } else {
            error(
                "It's look like your browser has no WebGL2 support.",
                true);
        }
    }

    function handleContextLoss(event) {
        event.preventDefault();
        K3D.disable();
        error('WEBGL Error', 'Context lost.', false);
    }

    canvas.addEventListener('webglcontextlost', handleContextLoss, false);

    self.renderer.removeContextLossListener = function () {
        canvas.removeEventListener('webglcontextlost', handleContextLoss);
    };

    const gl = self.renderer.getContext();

    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    console.log('K3D: (UNMASKED_VENDOR_WEBGL)', gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL));
    console.log('K3D: (UNMASKED_RENDERER_WEBGL)', gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL));
    console.log('K3D: (depth bits)', gl.getParameter(gl.DEPTH_BITS));
    console.log('K3D: (stencil bits)', gl.getParameter(gl.STENCIL_BITS));

    function standardRender(scene, camera, rt) {
        if (typeof (rt) === 'undefined') {
            rt = null;
        }

        self.renderer.setRenderTarget(rt);
        self.renderer.render(scene, camera);
    }

    function render() {
        const currentRenderMethod = standardRender;

        return new Promise((resolve) => {
            if (K3D.disabling) {
                return null;
            }

            const size = new THREE.Vector2();

            self.renderer.getSize(size);

            K3D.frameUpdateHandlers.before.forEach(handleListeners.bind(null, K3D, 'before'));

            K3D.refreshGrid();

            self.renderer.clippingPlanes = [];

            self.camera.updateMatrixWorld();

            self.renderer.clear();

            self.renderer.render(self.gridScene, self.camera);

            self.renderer.setViewport(size.x - self.axesHelper.width, 0, self.axesHelper.width, self.axesHelper.height);
            self.renderer.render(self.axesHelper.scene, self.axesHelper.camera);
            self.renderer.setViewport(0, 0, size.x, size.y);

            K3D.parameters.clippingPlanes.forEach((plane) => {
                self.renderer.clippingPlanes.push(new THREE.Plane(new THREE.Vector3().fromArray(plane), plane[3]));
            });

            K3D.dispatch(K3D.events.BEFORE_RENDER);

            let p = Promise.resolve();
            const originalControlsEnabledState = self.controls.enabled;

            function renderPass(x, y, width, height) {
                const chunkWidths = [];

                if (K3D.parameters.renderingSteps > 1) {
                    const s = width / K3D.parameters.renderingSteps;

                    for (let i = 0; i < K3D.parameters.renderingSteps; i++) {
                        const o1 = Math.round(i * s);
                        const o2 = Math.min(Math.round((i + 1) * s), width);
                        chunkWidths.push([o1, o2 - o1]);
                    }
                }

                if (K3D.parameters.renderingSteps > 1) {
                    self.controls.enabled = false;

                    if (self.controls.beforeRender) {
                        p = p.then(() => {
                            self.controls.beforeRender();
                        });
                    }

                    chunkWidths.forEach((c) => {
                        p = p.then(() => {
                            self.renderer.setViewport(x + c[0], y, c[1], height);
                            self.camera.setViewOffset(size.x, size.y, c[0], 0, c[1], size.y);

                            currentRenderMethod(self.scene, self.camera);
                        });

                        p = p.then(() => new Promise((chunkResolve) => {
                            setTimeout(chunkResolve, 50);
                        }));
                    });

                    if (self.controls.afterRender) {
                        p = p.then(() => {
                            self.controls.afterRender();
                        });
                    }
                } else {
                    p = p.then(() => {
                        if (self.controls.beforeRender) {
                            self.controls.beforeRender();
                        }

                        self.renderer.setViewport(x, y, width, height);
                        currentRenderMethod(self.scene, self.camera);

                        if (self.controls.afterRender) {
                            self.controls.afterRender();
                        }
                    });
                }
            }

            renderPass(0, 0, size.x, size.y);

            p = p.then(() => {
                self.controls.enabled = originalControlsEnabledState;

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

            return null;
        });
    }

    this.renderer.setClearColor(0, 0);
    this.renderer.autoClear = false;

    this.render = function (force) {
        K3D.labels = [];

        if (!K3D.autoRendering || force) {
            if (renderingPromise === null) {
                renderingPromise = render().then(() => {
                    renderingPromise = null;
                });

                return renderingPromise;
            }
            if (force) {
                renderingPromise = renderingPromise.then(render).then(() => {
                    renderingPromise = null;
                });
            }
        }

        return null;
    };

    this.renderOffScreen = function (width, height) {
        const chunkHeights = [];
        const chunkCount = Math.max(Math.min(128, K3D.parameters.renderingSteps), 1);
        const aaLevel = Math.max(Math.min(5, K3D.parameters.antialias), 0);
        const currentRenderMethod = standardRender;

        const s = height / chunkCount;

        const size = new THREE.Vector2();

        self.renderer.getSize(size);

        const scale = Math.max(width / size.x, height / size.y);

        for (let i = 0; i < chunkCount; i++) {
            const o1 = Math.round(i * s);
            const o2 = Math.min(Math.round((i + 1) * s), height);
            chunkHeights.push([o1, o2 - o1]);
        }

        const rt = new THREE.WebGLRenderTarget(width, Math.ceil(height / chunkCount), {
            type: THREE.FloatType,
        });

        const rtAxesHelper = new THREE.WebGLRenderTarget(self.axesHelper.width * scale,
            self.axesHelper.height * scale,
            {
                type: THREE.FloatType,
            });
        self.renderer.clippingPlanes = [];

        return getSSAAChunkedRender(self.renderer, self.axesHelper.scene, self.axesHelper.camera,
            rtAxesHelper, rtAxesHelper.width, rtAxesHelper.height, [[0, rtAxesHelper.height]],
            aaLevel, standardRender).then((result) => {
            const axesHelper = new Uint8ClampedArray(width * height * 4);

            for (let y = 0; y < rtAxesHelper.height; y++) {
                // fast row-copy
                axesHelper.set(
                    result.slice(y * rtAxesHelper.width * 4, (y + 1) * rtAxesHelper.width * 4),
                    (y * width + width - rtAxesHelper.width) * 4,
                );
            }

            return getSSAAChunkedRender(self.renderer, self.gridScene, self.camera,
                rt, width, height, [[0, height]], aaLevel, standardRender).then((grid) => {
                K3D.parameters.clippingPlanes.forEach((plane) => {
                    self.renderer.clippingPlanes.push(new THREE.Plane(new THREE.Vector3().fromArray(plane), plane[3]));
                });

                return getSSAAChunkedRender(self.renderer, self.scene, self.camera,
                    rt, width, height, chunkHeights,
                    aaLevel, currentRenderMethod).then((scene) => {
                    rt.dispose();
                    return [grid, scene, axesHelper];
                });
            });
        });
    };
};
