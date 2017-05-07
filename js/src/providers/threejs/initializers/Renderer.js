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

    var self = this, loop = false;

    self.renderer = new THREE.WebGLRenderer({
        antialias: K3D.parameters.antialias,
        preserveDrawingBuffer: false
    });

    function render() {
        if (K3D.disabling) {
            return void(0);
        }

        K3D.frameUpdateHandlers.before.forEach(handleListeners.bind(null, K3D, 'before'));

        self.renderer.render(self.scene, self.camera);

        K3D.frameUpdateHandlers.after.forEach(handleListeners.bind(null, K3D, 'after'));

        if (K3D.autoRendering) {
            requestAnimationFrame(render);
        } else {
            loop = false;
        }
    }

    this.renderer.setClearColor(0xffffff, 1);

    this.render = function () {
        if (!loop) {
            loop = true;
            render();
        }
    };
};
