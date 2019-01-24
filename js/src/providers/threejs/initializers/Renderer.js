'use strict';

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

    var self = this, loop = false,
        canvas = document.createElement('canvas'),
        context = canvas.getContext('webgl2'),
        gl, debugInfo;

    self.renderer = new THREE.WebGLRenderer({
        antialias: K3D.parameters.antialias,
        preserveDrawingBuffer: false,
        alpha: true,
        powerPreference: 'high-performance',
        canvas: canvas,
        context: context
    });

    gl = self.renderer.context;

    debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    console.log('K3D: (UNMASKED_VENDOR_WEBGL)', gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL));
    console.log('K3D: (UNMASKED_RENDERER_WEBGL)', gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL));

    function render() {
        if (K3D.disabling) {
            return void(0);
        }

        K3D.frameUpdateHandlers.before.forEach(handleListeners.bind(null, K3D, 'before'));

        K3D.refreshGrid();
        // self.renderer.clear();

        self.renderer.clippingPlanes = [];

        self.camera.updateMatrixWorld();
        self.renderer.render(self.gridScene, self.camera, null, true);

        K3D.parameters.clippingPlanes.forEach(function (plane) {
            self.renderer.clippingPlanes.push(new THREE.Plane(new THREE.Vector3().fromArray(plane), plane[3]));
        });

        K3D.dispatch(K3D.events.BEFORE_RENDER);

        self.renderer.render(self.scene, self.camera);

        K3D.frameUpdateHandlers.after.forEach(handleListeners.bind(null, K3D, 'after'));

        K3D.dispatch(K3D.events.RENDERED);

        if (K3D.autoRendering) {
            requestAnimationFrame(render);
        } else {
            loop = false;
        }
    }

    this.renderer.setClearColor(0, 0);
    this.renderer.autoClear = false;

    this.render = function () {
        if (!loop) {
            loop = true;
            render();
        }
    };

    this.renderOffScreen = function (width, height) {
        var ssaaRenderPass,
            rt = new THREE.WebGLRenderTarget(width, height),
            grid, scene;

        function getArrayFromRenderTarget(rt) {
            var array = new Uint8Array(width * height * 4);

            self.renderer.readRenderTargetPixels(rt, 0, 0, width, height, array);
            return new Uint8ClampedArray(array, width, height);
        }

        self.renderer.clearTarget(rt, true, true, true);
        self.renderer.clippingPlanes = [];

        ssaaRenderPass = new THREE.SSAARenderPass(self.gridScene, self.camera);
        ssaaRenderPass.sampleLevel = 4;
        ssaaRenderPass.setSize(width, height);
        ssaaRenderPass.render(self.renderer, rt, rt);
        ssaaRenderPass.dispose();
        grid = getArrayFromRenderTarget(rt);

        K3D.parameters.clippingPlanes.forEach(function (plane) {
            self.renderer.clippingPlanes.push(new THREE.Plane(new THREE.Vector3().fromArray(plane), plane[3]));
        });

        ssaaRenderPass = new THREE.SSAARenderPass(self.scene, self.camera);
        ssaaRenderPass.sampleLevel = 4;
        ssaaRenderPass.setSize(width, height);
        ssaaRenderPass.render(self.renderer, rt, rt);
        ssaaRenderPass.dispose();
        scene = getArrayFromRenderTarget(rt);


        rt.dispose();

        return [grid, scene];
    };
};
