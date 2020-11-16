'use strict';

var THREE = require('three'),
    cameraModes = require('./../../../core/lib/cameraMode').cameraModes,
    recalculateFrustum = require('./../helpers/Fn').recalculateFrustum;

function addEvents(self, K3D, controls) {
    controls.getCameraArray = function () {
        var r = [];

        self.controls.object.position.toArray(r);
        self.controls.target.toArray(r, 3);
        self.controls.object.up.toArray(r, 6);

        return r;
    };

    controls.addEventListener('change', function (event) {
        var camDistance , r = event.target.getCameraArray();

        recalculateFrustum(self.camera);

        K3D.dispatch(K3D.events.CAMERA_CHANGE, r);

        camDistance = (3.0 * 0.5) / Math.tan(
            THREE.Math.degToRad(K3D.parameters.camera_fov / 2.0)
        );

        self.axesHelper.camera.position.copy(
            self.camera.position.clone().sub(self.controls.target).normalize().multiplyScalar(camDistance)
        );
        self.axesHelper.camera.lookAt(0, 0, 0);
        self.axesHelper.camera.up.copy(self.camera.up);
    });

    controls.addEventListener('change', function () {
        if (K3D.frameUpdateHandlers.before.length === 0 && K3D.frameUpdateHandlers.after.length === 0) {
            self.render();
        }
    });
}

function createTrackballControls(self, K3D) {
    var controls = new THREE.TrackballControls(self.camera, self.renderer.domElement);

    controls.rotateSpeed = 1.0;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.8;
    controls.staticMoving = true;
    controls.dynamicDampingFactor = 0.1;

    addEvents(self, K3D, controls);

    return controls;
}

function createOrbitControls(self, K3D) {
    var controls = new THREE.OrbitControls(self.camera, self.renderer.domElement);

    controls.enableDamping = false;
    controls.dampingFactor = 0.1;
    controls.rotateSpeed = 0.5;
    controls.screenSpacePanning = false;
    controls.maxPolarAngle = Math.PI;
    controls.screenSpacePanning = true;

    addEvents(self, K3D, controls);

    return controls;
}

function createFlyControls(self, K3D) {
    var controls = new THREE.TrackballControls(self.camera, self.renderer.domElement);

    controls.rotateSpeed = 1.0;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 3.0;
    controls.flyMode = true;
    controls.staticMoving = true;
    controls.dynamicDampingFactor = 0.1;

    addEvents(self, K3D, controls);

    return controls;
}

function createControls(self, K3D) {
    if (K3D.parameters.cameraMode === cameraModes.trackball) {
        return createTrackballControls(self, K3D);
    } else if (K3D.parameters.cameraMode === cameraModes.orbit) {
        return createOrbitControls(self, K3D);
    } else if (K3D.parameters.cameraMode === cameraModes.fly) {
        return createFlyControls(self, K3D);
    }
}

/**
 * Canvas initializer for Three.js library
 * @this K3D.Core~world
 * @method Canvas
 * @memberof K3D.Providers.ThreeJS.Initializers
 */
module.exports = function (K3D) {

    var self = this, mouseCoordOnDown;

    function refresh() {
        if (K3D.disabling) {
            self.renderer.domElement.removeEventListener('mousemove', onDocumentMouseMove);
            self.renderer.domElement.removeEventListener('mousedown', onDocumentMouseDown);
            self.renderer.domElement.removeEventListener('mouseup', onDocumentMouseUp);
            self.controls.dispose();

            return;
        }

        self.controls.update();
        requestAnimationFrame(refresh);
    }

    function getCoordinate(event) {
        return {
            x: event.offsetX / K3D.getWorld().targetDOMNode.offsetWidth * 2 - 1,
            y: -event.offsetY / K3D.getWorld().targetDOMNode.offsetHeight * 2 + 1
        };
    }

    function onDocumentMouseDown(event) {
        mouseCoordOnDown = getCoordinate(event);
    }

    function onDocumentMouseUp(event) {
        var coordinate;

        coordinate = getCoordinate(event);

        if (mouseCoordOnDown.x === coordinate.x && mouseCoordOnDown.y === coordinate.y) {
            K3D.dispatch(K3D.events.MOUSE_CLICK, coordinate);
        }
    }

    function onDocumentMouseMove(event) {
        event.preventDefault();

        K3D.dispatch(K3D.events.MOUSE_MOVE, getCoordinate(event));
    }

    this.renderer.setSize(this.width, this.height);
    this.targetDOMNode.appendChild(this.renderer.domElement);

    this.renderer.domElement.addEventListener('mousemove', onDocumentMouseMove, false);
    this.renderer.domElement.addEventListener('mousedown', onDocumentMouseDown, false);
    this.renderer.domElement.addEventListener('mouseup', onDocumentMouseUp, false);

    this.controls = createControls(self, K3D);

    K3D.on(K3D.events.RESIZED, function () {
        if (self.controls.handleResize) {
            self.controls.handleResize();
        }
    });

    this.changeControls = function () {
        if (self.controls) {
            self.controls.dispose();
        }

        self.controls = createControls(self, K3D);
    };

    refresh();
};
