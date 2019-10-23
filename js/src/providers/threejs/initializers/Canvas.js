'use strict';

var THREE = require('three'),
    recalculateFrustum = require('./../helpers/Fn').recalculateFrustum;

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

    this.controls = new THREE.TrackballControls(this.camera, this.renderer.domElement);
    this.controls.rotateSpeed = 1.0;
    this.controls.zoomSpeed = 1.2;
    this.controls.panSpeed = 0.8;
    this.controls.staticMoving = true;
    this.controls.dynamicDampingFactor = 0.1;

    this.renderer.domElement.addEventListener('mousemove', onDocumentMouseMove, false);
    this.renderer.domElement.addEventListener('mousedown', onDocumentMouseDown, false);
    this.renderer.domElement.addEventListener('mouseup', onDocumentMouseUp, false);

    this.controls.getCameraArray = function () {
        var r = [];

        self.controls.object.position.toArray(r);
        self.controls.target.toArray(r, 3);
        self.controls.object.up.toArray(r, 6);

        return r;
    };

    this.controls.addEventListener('change', function (event) {
        var r = event.target.getCameraArray();

        recalculateFrustum(self.camera);

        K3D.dispatch(K3D.events.CAMERA_CHANGE, r);

        self.axesHelper.camera.position.copy(
            self.camera.position.clone().sub(self.controls.target).normalize().multiplyScalar(2.5)
        );
        self.axesHelper.camera.lookAt(0, 0, 0);
        self.axesHelper.camera.up.copy(self.camera.up);
    });

    K3D.on(K3D.events.RESIZED, function () {
        self.controls.handleResize();
    });

    refresh();
};
