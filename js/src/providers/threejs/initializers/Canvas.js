const THREE = require('three');
const { cameraModes } = require('../../../core/lib/cameraMode');
const { recalculateFrustum } = require('../helpers/Fn');

function addEvents(self, K3D, controls) {
    controls.getCameraArray = function () {
        const r = [];

        self.controls.object.position.toArray(r);
        self.controls.target.toArray(r, 3);
        self.controls.object.up.toArray(r, 6);

        return r;
    };

    controls.addEventListener('change', (event) => {
        const r = event.target.getCameraArray();

        recalculateFrustum(self.camera);

        K3D.dispatch(K3D.events.CAMERA_CHANGE, r);

        const camDistance = (3.0 * 0.5) / Math.tan(
            THREE.Math.degToRad(K3D.parameters.cameraFov / 2.0),
        );

        self.axesHelper.camera.position.copy(
            self.camera.position.clone().sub(self.controls.target).normalize().multiplyScalar(camDistance),
        );
        self.axesHelper.camera.lookAt(0, 0, 0);
        self.axesHelper.camera.up.copy(self.camera.up);
    });

    controls.addEventListener('change', () => {
        self.render();
    });
}

function createTrackballControls(self, K3D) {
    const controls = new THREE.TrackballControls(self.camera, self.renderer.domElement);

    controls.type = cameraModes.trackball;
    controls.rotateSpeed = K3D.parameters.cameraRotateSpeed;
    controls.zoomSpeed = K3D.parameters.cameraZoomSpeed;
    controls.panSpeed = K3D.parameters.cameraPanSpeed;

    if (K3D.parameters.cameraDampingFactor > 0.0) {
        controls.staticMoving = false;
        controls.dynamicDampingFactor = K3D.parameters.cameraDampingFactor;
    } else {
        controls.staticMoving = true;
    }

    addEvents(self, K3D, controls);

    return controls;
}

function createOrbitControls(self, K3D) {
    const controls = new THREE.OrbitControls(self.camera, self.renderer.domElement);

    controls.type = cameraModes.orbit;
    controls.rotateSpeed = K3D.parameters.cameraRotateSpeed;

    if (K3D.parameters.cameraDampingFactor > 0.0) {
        controls.enableDamping = true;
        controls.dampingFactor = K3D.parameters.cameraDampingFactor;
    } else {
        controls.enableDamping = false;
    }

    controls.screenSpacePanning = false;
    controls.maxPolarAngle = Math.PI;
    controls.screenSpacePanning = true;

    addEvents(self, K3D, controls);

    return controls;
}

function createFlyControls(self, K3D) {
    const controls = new THREE.TrackballControls(self.camera, self.renderer.domElement);

    controls.type = cameraModes.fly;
    controls.rotateSpeed = K3D.parameters.cameraRotateSpeed;
    controls.zoomSpeed = K3D.parameters.cameraZoomSpeed;
    controls.panSpeed = K3D.parameters.cameraPanSpeed;
    controls.flyMode = true;

    if (K3D.parameters.cameraDampingFactor > 0.0) {
        controls.staticMoving = false;
        controls.dynamicDampingFactor = K3D.parameters.cameraDampingFactor;
    } else {
        controls.staticMoving = true;
    }

    addEvents(self, K3D, controls);

    return controls;
}

function createControls(self, K3D) {
    if (K3D.parameters.cameraMode === cameraModes.trackball) {
        return createTrackballControls(self, K3D);
    }
    if (K3D.parameters.cameraMode === cameraModes.orbit) {
        return createOrbitControls(self, K3D);
    }
    if (K3D.parameters.cameraMode === cameraModes.fly) {
        return createFlyControls(self, K3D);
    }

    return null;
}

/**
 * Canvas initializer for Three.js library
 * @this K3D.Core~world
 * @method Canvas
 * @memberof K3D.Providers.ThreeJS.Initializers
 */
module.exports = function (K3D) {
    const self = this;
    let mouseCoordOnDown;
    let lastFrame = (new Date()).getTime();

    function refresh() {
        let currentFrame = (new Date()).getTime();

        K3D.frameInterval = currentFrame - lastFrame;
        lastFrame = currentFrame;

        const { targetDOMNode } = K3D.getWorld();

        if (!targetDOMNode.ownerDocument.contains(targetDOMNode)) {
            K3D.disable();
        }

        if (K3D.disabling) {
            self.renderer.domElement.removeEventListener('pointermove', onDocumentMouseMove);
            self.renderer.domElement.removeEventListener('pointerdown', onDocumentMouseDown);
            self.renderer.domElement.removeEventListener('pointerup', onDocumentMouseUp);
            self.controls.dispose();

            return;
        }

        self.controls.update();
        requestAnimationFrame(refresh);
    }

    function getCoordinate(event) {
        return {
            x: (event.offsetX / K3D.getWorld().targetDOMNode.offsetWidth) * 2 - 1,
            y: (-event.offsetY / K3D.getWorld().targetDOMNode.offsetHeight) * 2 + 1,
        };
    }

    function onDocumentMouseDown(event) {
        mouseCoordOnDown = getCoordinate(event);
    }

    function onDocumentMouseUp(event) {
        const coordinate = getCoordinate(event);

        if (mouseCoordOnDown.x === coordinate.x && mouseCoordOnDown.y === coordinate.y) {
            K3D.dispatch(K3D.events.MOUSE_CLICK, coordinate);
        }
    }

    function onDocumentMouseMove(event) {
        K3D.dispatch(K3D.events.MOUSE_MOVE, getCoordinate(event));
    }

    this.renderer.setSize(this.width, this.height);
    this.targetDOMNode.appendChild(this.renderer.domElement);

    this.renderer.domElement.addEventListener('pointermove', onDocumentMouseMove, false);
    this.renderer.domElement.addEventListener('pointerdown', onDocumentMouseDown, false);
    this.renderer.domElement.addEventListener('pointerup', onDocumentMouseUp, false);

    this.controls = createControls(self, K3D);

    K3D.on(K3D.events.RESIZED, () => {
        if (self.controls.handleResize) {
            self.controls.handleResize();
        }
    });

    this.changeControls = function (force) {
        if (self.controls.type === K3D.parameters.cameraMode && !force) {
            return;
        }

        if (self.controls) {
            self.controls.dispose();
        }

        self.controls = createControls(self, K3D);
    };

    refresh();
};
