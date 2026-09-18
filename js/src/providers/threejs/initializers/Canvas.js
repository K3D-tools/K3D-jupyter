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

        const camDistance = (3.0 * 0.5) / Math.tan(THREE.MathUtils.degToRad(K3D.parameters.cameraFov / 2.0));

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
    controls.zoomSpeed = K3D.parameters.cameraZoomSpeed;
    controls.panSpeed = K3D.parameters.cameraPanSpeed;

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
    const controls = new THREE.TrackballControls(self.camera, self.renderer.domElement, K3D);

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

function createSliceControls(self, K3D) {
    const controls = new THREE.SliceControls(self.camera, self.renderer.domElement, K3D);

    controls.type = cameraModes.sliceViewer;

    addEvents(self, K3D, controls);

    return controls;
}

function createVolumeSideControls(self, K3D) {
    const controls = new THREE.VolumeSidesControls(self.camera, self.renderer.domElement, K3D);

    controls.type = cameraModes.volumeSides;

    addEvents(self, K3D, controls);

    return controls;
}

// both families of flags, since Core writes both: Trackball reads no*, Orbit enable*
function applyCameraLock(controls, K3D) {
    controls.noRotate = K3D.parameters.cameraNoRotate;
    controls.noZoom = K3D.parameters.cameraNoZoom;
    controls.noPan = K3D.parameters.cameraNoPan;
    controls.enableRotate = !K3D.parameters.cameraNoRotate;
    controls.enableZoom = !K3D.parameters.cameraNoZoom;
    controls.enablePan = !K3D.parameters.cameraNoPan;
}

function createControls(self, K3D) {
    let controls = null;

    if (K3D.parameters.cameraMode === cameraModes.trackball) {
        controls = createTrackballControls(self, K3D);
    } else if (K3D.parameters.cameraMode === cameraModes.orbit) {
        controls = createOrbitControls(self, K3D);
    } else if (K3D.parameters.cameraMode === cameraModes.fly) {
        controls = createFlyControls(self, K3D);
    } else if (K3D.parameters.cameraMode === cameraModes.sliceViewer) {
        controls = createSliceControls(self, K3D);
    } else if (K3D.parameters.cameraMode === cameraModes.volumeSides) {
        controls = createVolumeSideControls(self, K3D);
    } else {
        // no controls at all means a dead canvas and an error only in the browser console
        console.warn(`K3D: unknown camera_mode '${K3D.parameters.cameraMode}', using trackball`);
        K3D.parameters.cameraMode = cameraModes.trackball;
        controls = createTrackballControls(self, K3D);
    }

    if (controls !== null) {
        // fresh controls know nothing of the lock in force
        applyCameraLock(controls, K3D);
    }

    return controls;
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
    let wasAttached = false;
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

        const { targetDOMNode } = K3D.getWorld();
        const attached = targetDOMNode.ownerDocument.contains(targetDOMNode);

        // refresh() runs once from this initializer, so a bare "not in the document" test would
        // tear down a host that inserts its subtree later, Panel among them.
        if (attached) {
            wasAttached = true;
        } else if (wasAttached) {
            K3D.disable();
        }

        if (K3D.disabling) {
            self.renderer.domElement.removeEventListener('pointermove', onDocumentMouseMove);
            self.renderer.domElement.removeEventListener('pointerdown', onDocumentMouseDown);
            self.renderer.domElement.removeEventListener('pointerup', onDocumentMouseUp);
            self.renderer.domElement.removeEventListener('pointerleave', onDocumentMouseLeave);
            window.removeEventListener('visibilitychange', onVisibilityChange);
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

        if (mouseCoordOnDown
            && mouseCoordOnDown.x === coordinate.x && mouseCoordOnDown.y === coordinate.y) {
            K3D.dispatch(K3D.events.MOUSE_CLICK, coordinate);
        }
    }

    function onDocumentMouseMove(event) {
        K3D.dispatch(K3D.events.MOUSE_MOVE, getCoordinate(event));
    }

    function onDocumentMouseLeave() {
        K3D.dispatch(K3D.events.MOUSE_LEAVE);
    }

    this.renderer.setSize(this.width, this.height);
    this.targetDOMNode.appendChild(this.renderer.domElement);

    this.renderer.domElement.addEventListener('pointermove', onDocumentMouseMove, false);
    this.renderer.domElement.addEventListener('pointerdown', onDocumentMouseDown, false);
    this.renderer.domElement.addEventListener('pointerup', onDocumentMouseUp, false);
    this.renderer.domElement.addEventListener('pointerleave', onDocumentMouseLeave, false);

    this.controls = createControls(self, K3D);

    K3D.on(K3D.events.RESIZED, () => {
        if (self.controls.handleResize) {
            self.controls.handleResize();
        }
    });

    function onVisibilityChange() {
        lastFrameTime = null;
    }

    window.addEventListener('visibilitychange', onVisibilityChange);

    this.changeControls = function (force) {
        if (self.controls.type === K3D.parameters.cameraMode && !force) {
            return;
        }

        // the camera keeps its position, so fresh controls looking at the origin swing the view
        const target = (self.controls && self.controls.target)
            ? self.controls.target.clone() : null;

        if (self.controls) {
            self.controls.dispose();
        }

        self.controls = createControls(self, K3D);

        if (target && self.controls.target) {
            self.controls.target.copy(target);
        }
    };

    refresh();
};
