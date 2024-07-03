const THREE = require('three');
const {cameraModes} = require('../../../core/lib/cameraMode');
const {recalculateFrustum} = require('../helpers/Fn');

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

        const camDistance = (3.0 * 0.5) / Math.tan(THREE.Math.degToRad(K3D.parameters.cameraFov / 2.0));

        self.axesHelper.camera.position.copy(self.camera.position.clone().sub(self.controls.target).normalize()
            .multiplyScalar(camDistance));
        self.axesHelper.camera.lookAt(0, 0, 0);
        self.axesHelper.camera.up.copy(self.camera.up);
    });

    controls.addEventListener('change', () => {
        self.render();
    });
}

function createTrackballControls(self, K3D) {
    const controls = new THREE.TrackballControls(self.camera, self.renderer.domElement, K3D);

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
    const controls = new THREE.OrbitControls(self.camera, self.renderer.domElement, K3D);

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
    let lastFrameTime = null;
    const intervals = new Float32Array(64);
    let intervalsPtr = 0;
    let qualityFactor = 1.0;

    function changeQuality(quality) {
        self.renderer.setPixelRatio(window.devicePixelRatio * quality);
    }

    function guessQualityFactor(time) {
        if (time < 1000.0 / (2.0 * K3D.parameters.minimumFps)) {
            return Math.min(qualityFactor * 1.5, 1);
        }

        return qualityFactor / Math.min((1.25 * time) / (1000.0 / K3D.parameters.minimumFps), 5);
    }

    function refresh(time, skipFrameCount) {
        if (K3D.parameters.minimumFps > 0) {
            if (!time) {
                // fired manually - we need correct parameters
                requestAnimationFrame(refresh);
                return;
            }

            if (lastFrameTime === null) {
                lastFrameTime = time;
                requestAnimationFrame(refresh);
                return;
            }

            const currentFrame = time;

            K3D.frameInterval = currentFrame - lastFrameTime;

            lastFrameTime = currentFrame;

            if (skipFrameCount > 0 || K3D.heavyOperationAsync || K3D.heavyOperationSync) {
                if (K3D.heavyOperationSync) {
                    skipFrameCount = 16;
                    K3D.heavyOperationSync = false;
                }

                self.controls.update();
                requestAnimationFrame((t) => {
                    refresh(t, skipFrameCount - 1);
                });
                return;
            }

            // adaptative resolution
            intervals[intervalsPtr] = K3D.frameInterval;
            intervalsPtr = (intervalsPtr + 1) % intervals.length;
            const longAverageTime = intervals.reduce((a, b) => a + b, 0) / intervals.length;

            let shortAverageTime = 0;
            for (let i = intervalsPtr - 8 + intervals.length; i < intervalsPtr + intervals.length; i++) {
                shortAverageTime += intervals[i % intervals.length];
            }
            shortAverageTime /= 8;

            const oldQualityFactor = qualityFactor;

            if (K3D.frameInterval > (2.0 * 1000.0) / K3D.parameters.minimumFps) {
                qualityFactor = guessQualityFactor(K3D.frameInterval);
            } else if (shortAverageTime > 1000.0 / K3D.parameters.minimumFps) {
                qualityFactor = guessQualityFactor(shortAverageTime);
            } else if (longAverageTime < 1000.0 / (2.0 * K3D.parameters.minimumFps)) {
                qualityFactor = guessQualityFactor(longAverageTime);
            }

            if (oldQualityFactor !== qualityFactor) {
                console.log('qualityFactor', qualityFactor);

                changeQuality(qualityFactor);
                K3D.render();
                intervals.fill(1000.0 / (1.5 * K3D.parameters.minimumFps));

                requestAnimationFrame((t) => {
                    refresh(t, 16);
                });

                return;
            }
        }

        const {targetDOMNode} = K3D.getWorld();

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

    window.addEventListener('visibilitychange', () => {
        lastFrameTime = null;
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
