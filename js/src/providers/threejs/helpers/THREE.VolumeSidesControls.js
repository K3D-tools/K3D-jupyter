const objectsGUIprovider = require('../../../core/lib/objectsGUIprovider');
const _ = require('../../../lodash');

module.exports = function (THREE) {
    THREE.VolumeSidesControls = function (object, domElement, K3D) {
        const scope = this;

        function getSliceJson() {
            const objectId = K3D.parameters.sliceViewerObjectId;
            const json = K3D.getWorld().ObjectsListJson[objectId];

            return json;
        }

        const _this = this;
        const cameraTrackball = object.clone();
        const trackBall = new THREE.TrackballControls(cameraTrackball, domElement, K3D);
        const STATE = {
            NONE: -1, LEVELS: 0, ZOOM: 1, PAN: 2, TOUCH_SLICES: 3, TOUCH_PAN_ZOOM: 4, TOUCH_LEVELS: 5,
        };

        let currentWindow;
        let currentDocument;

        if (domElement !== undefined) {
            currentWindow = domElement.ownerDocument.defaultView || domElement.ownerDocument.parentWindow;
            this.domElement = domElement;
            currentDocument = domElement.ownerDocument;
        } else {
            currentWindow = window;
            this.domElement = currentWindow.document;
            currentDocument = currentWindow.document;
        }

        this.object = object;
        this.domElement.style.touchAction = 'none'; // disable touch scroll
        let volumeSlice = null;
        let lastMode = -1;

        // API

        this.enabled = true;

        this.screen = {
            left: 0, top: 0, width: 0, height: 0,
        };
        const _pointers = [];
        const _pointerPositions = {};

        this.minDistance = 0;
        this.maxDistance = Infinity;

        // internals
        this.target = new THREE.Vector3();

        const EPS = 0.0000000001;

        const lastSlices = new THREE.Vector3(-1, -1, -1);

        let _state = STATE.NONE;

        const _mouseCurrent = new THREE.Vector2();
        const _mouseLast = new THREE.Vector2();

        const _zoomPanCurrent = {
            x: new THREE.Vector3(),
            y: new THREE.Vector3(),
            z: new THREE.Vector3(),
        };
        const _zoomPanLast = new THREE.Vector3();

        let _touchZoomDistanceStart = 0;
        let _touchZoomDistanceEnd = 0;
        let _wheelDelta = 0;

        // events

        const changeEvent = { type: 'change' };
        const startEvent = { type: 'start' };
        const endEvent = { type: 'end' };

        // methods

        this.handleResize = function () {
            if (this.domElement === currentDocument) {
                this.screen.left = 0;
                this.screen.top = 0;
                this.screen.width = currentWindow.innerWidth;
                this.screen.height = currentWindow.innerHeight;
            } else {
                const box = this.domElement.getBoundingClientRect();
                // adjustments come from similar code in the jquery offset() function
                const d = this.domElement.ownerDocument.documentElement;
                this.screen.left = box.left + currentWindow.pageXOffset - d.clientLeft;
                this.screen.top = box.top + currentWindow.pageYOffset - d.clientTop;
                this.screen.width = box.width;
                this.screen.height = box.height;
            }
        };

        this.handleEvent = function (event) {
            if (typeof this[event.type] === 'function') {
                this[event.type](event);
            }
        };

        this.beforeRender = function (scene) {
            const json = getSliceJson();

            if (!json) {
                return;
            }

            const objectProvider = K3D.Provider.Objects[json.type];

            if (!volumeSlice) {
                volumeSlice = K3D.getObjectById(K3D.parameters.sliceViewerObjectId);

                if (!volumeSlice) {
                    return;
                }

                if (!trackBall.initialized) {
                    trackBall.target = volumeSlice.position.clone();
                    const objectBoundingBox = volumeSlice.geometry.boundingBox.clone();
                    objectBoundingBox.applyMatrix4(volumeSlice.matrixWorld);
                    const objectBoundingSphere = objectBoundingBox.getBoundingSphere(new THREE.Sphere());

                    cameraTrackball.position.subVectors(
                        objectBoundingSphere.center,
                        (new THREE.Vector3(-1, 2, -1)).setLength(objectBoundingSphere.radius * 1.5),
                    );
                    trackBall.update();
                    trackBall.initialized = true;
                }
            }

            const jsons = K3D.getWorld().ObjectsListJson;

            Object.keys(jsons).forEach((id) => {
                id = parseInt(id, 10);
                const obj = K3D.getObjectById(id);

                if (obj) {
                    obj.originalVisible = obj.visible;

                    if (scene < 3) {
                        if (id !== json.id && (!jsons[id].slice_planes || jsons[id].slice_planes.length === 0)) {
                            obj.visible = false;
                        }
                    }

                    if (scene === 3) {
                        if (K3D.parameters.hiddenObjectIds.indexOf(id) !== -1) {
                            obj.visible = false;
                        }
                    }
                }
            });

            K3D.parameters.sliceViewerMaskObjectIds.forEach((objId) => {
                const jsonMask = K3D.getWorld().ObjectsListJson[objId];
                jsonMask.originalSlicePlanes = jsonMask.slice_planes;
            });

            switch (scene) {
                case 0:
                    _this.prepareView(json, 'z');
                    objectProvider.update(json, { slice_x: -1, slice_y: -1, slice_z: json.slice_z }, volumeSlice, K3D);
                    break;
                case 1:
                    _this.prepareView(json, 'y');
                    objectProvider.update(json, { slice_x: -1, slice_y: json.slice_y, slice_z: -1 }, volumeSlice, K3D);
                    break;
                case 2:
                    _this.prepareView(json, 'x');
                    objectProvider.update(json, { slice_x: json.slice_x, slice_y: -1, slice_z: -1 }, volumeSlice, K3D);
                    break;
                case 3:
                    cameraTrackball.far = _this.object.far;
                    cameraTrackball.near = _this.object.near;
                    cameraTrackball.aspect = _this.object.aspect;
                    cameraTrackball.updateProjectionMatrix();

                    _this.target.copy(trackBall.target);
                    _this.object.copy(trackBall.object, false);
                    break;
                default:
                    break;
            }
        };

        this.afterRender = function (scene) {
            const json = getSliceJson();
            let objectProvider;

            if (!json || !volumeSlice) {
                return;
            }

            const jsons = K3D.getWorld().ObjectsListJson;

            Object.keys(jsons).forEach((id) => {
                id = parseInt(id, 10);
                const obj = K3D.getObjectById(id);

                if (obj && typeof (obj.originalVisible) !== 'undefined') {
                    obj.visible = obj.originalVisible;
                    delete obj.originalVisible;
                }
            });

            K3D.parameters.sliceViewerMaskObjectIds.forEach((objId) => {
                const jsonMask = K3D.getWorld().ObjectsListJson[objId];
                jsonMask.slice_planes = jsonMask.originalSlicePlanes;

                const obj = K3D.getObjectById(objId);

                if (obj) {
                    objectProvider = K3D.Provider.Objects[jsonMask.type];
                    objectProvider.update(jsonMask, { slice_planes: jsonMask.slice_planes }, obj, K3D);
                }

                delete jsonMask.originalSlicePlanes;
            });

            if (scene === 2) {
                objectProvider = K3D.Provider.Objects[json.type];
                objectProvider.update(json, {
                    slice_x: json.slice_x,
                    slice_y: json.slice_y,
                    slice_z: json.slice_z,
                }, volumeSlice, K3D);
            }

            if (scene === 3) {
                volumeSlice = null;
            }
        };

        const getMouseOnScreen = (function () {
            const vector = new THREE.Vector2();

            return function (x, y) {
                vector.set(
                    x / _this.screen.width,
                    y / _this.screen.height,
                );

                return vector;
            };
        }());

        this.changeSlice = function (json, change) {
            let shape = Array.isArray(json.volume) ? json.volume[0].shape : json.volume.shape;

            if (_mouseCurrent.x < 0.5 && _mouseCurrent.y < 0.5) {
                json.slice_z += change;
                json.slice_z = Math.max(-1, Math.min(json.slice_z, shape[0] - 1));
            }

            if (_mouseCurrent.x < 0.5 && _mouseCurrent.y > 0.5) {
                json.slice_y += change;
                json.slice_y = Math.max(-1, Math.min(json.slice_y, shape[1] - 1));
            }

            if (_mouseCurrent.x > 0.5 && _mouseCurrent.y < 0.5) {
                json.slice_x += change;
                json.slice_x = Math.max(-1, Math.min(json.slice_x, shape[2] - 1));
            }

            return change !== 0;
        };

        this.changeLevels = function (json) {
            const change = _mouseCurrent.clone().sub(_mouseLast);
            const mag = (json.color_range[1] - json.color_range[0]) * 0.5;

            if (change.length() > 0) {
                if (Math.abs(change.y) > EPS) {
                    json.color_range[0] += change.y * mag;
                    json.color_range[1] += change.y * mag;
                }

                if (Math.abs(change.x) > EPS) {
                    json.color_range[0] += change.x * mag;
                    json.color_range[1] -= change.x * mag;

                    if (json.color_range[0] >= json.color_range[1]) {
                        json.color_range[0] = (json.color_range[0] + json.color_range[1]) * 0.5 - 1;
                        json.color_range[1] = json.color_range[0] + 2;
                    }
                }

                objectsGUIprovider.changeParameter(K3D, json, 'color_range', json.color_range, true);

                return true;
            }

            return false;
        };

        this.changeZoom = function (json) {
            const change = _mouseCurrent.clone().sub(_mouseLast);

            if (change.length() > 0) {
                if (Math.abs(change.y) > EPS) {
                    _zoomPanCurrent.x.z += change.y;
                    _zoomPanCurrent.y.z += change.y;
                    _zoomPanCurrent.z.z += change.y;
                }

                if (Math.abs(change.x) > EPS) {
                    _this.changeSlice(json, Math.round(change.x * 500));
                }

                return true;
            }

            return false;
        };

        this.changePan = function () {
            const change = _mouseCurrent.clone().sub(_mouseLast);
            let bonus = 0;

            if (_touchZoomDistanceEnd !== _touchZoomDistanceStart) {
                bonus = (_touchZoomDistanceStart - _touchZoomDistanceEnd) * 0.0005;
            }

            if (change.length() > 0) {
                if (_mouseCurrent.x < 0.5 && _mouseCurrent.y < 0.5) {
                    _zoomPanCurrent.z.x += change.x * 2;
                    _zoomPanCurrent.z.y += change.y * 2;
                    _zoomPanCurrent.z.z += bonus;
                }

                if (_mouseCurrent.x < 0.5 && _mouseCurrent.y > 0.5) {
                    _zoomPanCurrent.y.x += change.x * 2;
                    _zoomPanCurrent.y.y += change.y * 2;
                    _zoomPanCurrent.y.z += bonus;
                }

                if (_mouseCurrent.x > 0.5 && _mouseCurrent.y < 0.5) {
                    _zoomPanCurrent.x.x += change.x * 2;
                    _zoomPanCurrent.x.y += change.y * 2;
                    _zoomPanCurrent.x.z += bonus;
                }

                return true;
            }

            return false;
        };

        this.prepareView = function (json, axis, onlyReturn) {
            const obj = K3D.getObjectById(json.id);
            let up = new THREE.Vector3();
            const right = new THREE.Vector3();
            let ray = new THREE.Vector3();
            const axisToIndex = {
                x: [0, 2, 1],
                y: [1, 2, 0],
                z: [2, 1, 0],
            };
            const axisToShape = {
                x: 2,
                y: 1,
                z: 0,
            };
            let sliceDistance = json[`slice_${axis}`];
            let shape = Array.isArray(json.volume) ? json.volume[0].shape : json.volume.shape;

            if (!obj) {
                return {};
            }

            ray.setFromMatrixColumn(obj.matrixWorld, axisToIndex[axis][0]);
            up.setFromMatrixColumn(obj.matrixWorld, axisToIndex[axis][1]);
            right.setFromMatrixColumn(obj.matrixWorld, axisToIndex[axis][2]);

            if (axis === 'x') {
                // up = up.negate();
                ray = ray.negate();
                sliceDistance = shape[2] - 1 - json.slice_x;
            }

            if (axis === 'y') {
                // up = up.negate();
                ray = ray.negate();
                sliceDistance = shape[1] - 1 - json.slice_y;
            }

            if (axis === 'z') {
                up = up.negate();
                ray = ray.negate();
                sliceDistance = shape[0] - 1 - json.slice_z;
            }

            const slicePosition = obj.position.clone().sub(ray.multiplyScalar(0.5)).add(
                ray.clone().multiplyScalar(
                    (2.0 * sliceDistance + 0.1 + 1.0) / (shape[axisToShape[axis]]),
                ),
            );

            slicePosition.add(
                up.clone().multiplyScalar(_zoomPanCurrent[axis].y),
            ).add(
                right.clone().multiplyScalar(-_zoomPanCurrent[axis].x),
            );

            sliceDistance = (up.length() * 0.5) / Math.tan(
                THREE.Math.degToRad(K3D.parameters.cameraFov / 2.0),
            );
            const camDistance = sliceDistance * (1.0 + _zoomPanCurrent[axis].z * 0.5);

            if (onlyReturn) {
                return {
                    ray,
                    slicePosition,
                };
            }

            _this.object.position.copy(slicePosition.clone().add(
                ray.clone().normalize().multiplyScalar(camDistance),
            ));
            _this.object.up.copy(up.clone().normalize());
            _this.target.copy(slicePosition);
            _this.object.lookAt(_this.target);
            _this.object.updateProjectionMatrix();
            _this.object.updateWorldMatrix();

            const index = ['x', 'y', 'z'].indexOf(axis);
            const objectProvider = K3D.Provider.Objects.Mesh;

            K3D.parameters.sliceViewerMaskObjectIds.forEach((objId) => {
                const o = K3D.getObjectById(objId);
                const jsonConfig = K3D.getWorld().ObjectsListJson[objId];
                const slicePlane = jsonConfig.slice_planes;

                if (o) {
                    objectProvider.update(jsonConfig, { slice_planes: [slicePlane[index]] }, o, K3D);
                }
            });

            return null;
        };

        this.update = function () {
            const json = getSliceJson();
            let change = false;

            if (typeof (json) === 'undefined') {
                return;
            }

            if (!(_mouseCurrent.x > 0.5 && _mouseCurrent.y > 0.5)) {
                trackBall._pointers = [];
                trackBall._pointerPositions = {};

                if (lastMode === null) {
                    lastMode = 1;
                }

                change |= _this.changeSlice(json, Math.sign(_wheelDelta));

                if (lastMode === 1) {
                    if (_state === STATE.LEVELS) {
                        change |= _this.changeLevels(json);
                    }

                    if (_state === STATE.ZOOM) {
                        change |= _this.changeZoom(json);
                    }

                    if (_state === STATE.PAN) {
                        change |= _this.changePan();
                    }

                    if (_state === STATE.TOUCH_SLICES) {
                        change |= _this.changeSlice(json, Math.sign(_mouseCurrent.y - _mouseLast.y));
                    }

                    if (_state === STATE.TOUCH_PAN_ZOOM) {
                        change |= _this.changePan();
                    }

                    if (_state === STATE.TOUCH_LEVELS) {
                        change |= _this.changeLevels(json);
                    }
                }

                // update slice?
                const planes = [];
                let update = false;
                ['x', 'y', 'z'].forEach((axis) => {
                    if (json[`slice_${axis}`] !== lastSlices[axis]) {
                        const changes = {};
                        update = true;

                        changes[`slice_${axis}`] = json[`slice_${axis}`];

                        objectsGUIprovider.update(K3D, json, K3D.GUI.objects, changes);
                        objectsGUIprovider.changeParameter(K3D, json, `slice_${axis}`, json[`slice_${axis}`], true);
                        lastSlices[axis] = json[`slice_${axis}`];
                    }
                });

                if (K3D.parameters.sliceViewerMaskObjectIds.length > 0 && update) {
                    ['x', 'y', 'z'].forEach((axis) => {
                        const data = _this.prepareView(json, axis, true);

                        planes.push(new THREE.Plane().setFromNormalAndCoplanarPoint(
                            data.ray.clone().normalize(),
                            data.slicePosition,
                        ));
                    });

                    K3D.parameters.sliceViewerMaskObjectIds.forEach((objId) => {
                        const sliceJson = K3D.getWorld().ObjectsListJson[objId];
                        const sliceChange = {};

                        if (sliceJson.opacity === 1.0) {
                            sliceChange.opacity = 0.95;
                            sliceJson.opacity = 0.95;
                        }

                        sliceChange.slice_planes = planes.map(
                            (plane) => [plane.normal.x, plane.normal.y, plane.normal.z, plane.constant],
                        );
                        sliceJson.slice_planes = sliceChange.slice_planes;

                        if (sliceJson.visible) {
                            K3D.reload(sliceJson, sliceChange, true);
                        }
                    });
                }

                // discard change
                _wheelDelta = 0;
                _mouseLast.copy(_mouseCurrent);

                if (change) {
                    _this.dispatchEvent(changeEvent);
                }
            } else {
                if (lastMode === null) {
                    lastMode = 2;
                }

                if (lastMode === 2 || lastMode === -1) {
                    trackBall.update();
                }
            }
        };

        this.reslice = function () {
            lastSlices.set(-1, -1, -1);
        };

        this.reset = function () {
            _state = STATE.NONE;

            _zoomPanCurrent.x.set(0.0, 0.0, 0.0);
            _zoomPanCurrent.y.set(0.0, 0.0, 0.0);
            _zoomPanCurrent.z.set(0.0, 0.0, 0.0);

            _zoomPanLast.set(0.0, 0.0, 0.0);

            _this.dispatchEvent(changeEvent);
        };

        function addPointer(event) {
            _pointers.push(event);
        }

        function removePointer(event) {
            delete _pointerPositions[event.pointerId];

            for (let i = 0; i < _pointers.length; i++) {
                if (_pointers[i].pointerId == event.pointerId) {
                    _pointers.splice(i, 1);
                    return;
                }
            }
        }

        function trackPointer(event) {
            let position = _pointerPositions[event.pointerId];

            if (position === undefined) {
                position = new THREE.Vector2();
                _pointerPositions[event.pointerId] = position;
            }

            position.set(event.offsetX, event.offsetY);
        }

        function getSecondPointerPosition(event) {
            const pointer = (event.pointerId === _pointers[0].pointerId) ? _pointers[1] : _pointers[0];

            return _pointerPositions[pointer.pointerId];
        }

        // listeners

        function onPointerDown(event) {
            if (_pointers.length === 0) {
                scope.domElement.setPointerCapture(event.pointerId);
            }
            lastMode = null;

            addPointer(event);

            if (event.pointerType === 'touch') {
                onTouchStart(event);
            } else {
                onMouseDown(event);
            }
        }

        function onPointerMove(event) {
            if (event.pointerType === 'touch') {
                onTouchMove(event);
            } else {
                onMouseMove(event);
            }
        }

        function onPointerUp(event) {
            if (event.pointerType === 'touch') {
                onTouchEnd();
            } else {
                onMouseUp();
            }
            removePointer(event);
            lastMode = -1;

            if (_pointers.length === 0) {
                scope.domElement.releasePointerCapture(event.pointerId);
            }
        }

        function onPointerCancel(event) {
            removePointer(event);
        }

        function onTouchStart(event) {
            trackPointer(event);

            switch (_pointers.length) {
                case 1:
                    _state = STATE.TOUCH_SLICES;
                    _mouseCurrent.copy(getMouseOnScreen(_pointers[0].offsetX, _pointers[0].offsetY));
                    _mouseLast.copy(_mouseCurrent);
                    break;

                default:

                    const position = getSecondPointerPosition(event);

                    const dx = event.offsetX - position.x;
                    const dy = event.offsetY - position.y;
                    _touchZoomDistanceEnd = _touchZoomDistanceStart = Math.sqrt(dx * dx + dy * dy);

                    const x = (event.offsetX + position.x) / 2;
                    const y = (event.offsetY + position.y) / 2;
                    _mouseCurrent.copy(getMouseOnScreen(x, y));
                    _mouseLast.copy(_mouseCurrent);

                    if (_touchZoomDistanceStart > Math.min(_this.screen.width, _this.screen.height) / 3.0) {
                        _state = STATE.TOUCH_PAN_ZOOM;
                    } else {
                        _state = STATE.TOUCH_LEVELS;
                    }

                    break;
            }

            scope.dispatchEvent(startEvent);
        }

        function onTouchMove(event) {
            trackPointer(event);

            switch (_pointers.length) {
                case 1:
                    _mouseLast.copy(_mouseCurrent);
                    _mouseCurrent.copy(getMouseOnScreen(event.offsetX, event.offsetY));
                    break;
                case 2:
                    _mouseLast.copy(_mouseCurrent);

                    const position = getSecondPointerPosition(event);

                    const dx = event.offsetX - position.x;
                    const dy = event.offsetY - position.y;

                    _touchZoomDistanceEnd = Math.sqrt(dx * dx + dy * dy);

                    const x = (event.offsetX + position.x) / 2;
                    const y = (event.offsetY + position.y) / 2;
                    _mouseCurrent.copy(getMouseOnScreen(x, y));
                    break;
                default: // 3 or more
                    _mouseLast.copy(_mouseCurrent);
                    _mouseCurrent.copy(getMouseOnScreen(_pointers[0].offsetX, _pointers[0].offsetY));
                    break;
            }
        }

        function onTouchEnd() {
            _state = STATE.NONE;
            _this.dispatchEvent(endEvent);
        }

        function onMouseDown(event) {
            event.preventDefault();

            if (_state === STATE.NONE) {
                _state = event.button;
            }

            _mouseCurrent.copy(getMouseOnScreen(event.offsetX, event.offsetY));
            _mouseLast.copy(_mouseCurrent);
            _this.dispatchEvent(startEvent);
        }

        function onMouseMove(event) {
            _mouseCurrent.copy(getMouseOnScreen(event.offsetX, event.offsetY));
        }

        function onMouseUp() {
            _state = STATE.NONE;
            _this.dispatchEvent(endEvent);
        }

        function onMouseWheel(event) {
            event.preventDefault();

            switch (event.deltaMode) {
                case 2:
                    // Zoom in pages
                    _wheelDelta = event.deltaY * 0.025;
                    break;

                case 1:
                    // Zoom in lines
                    _wheelDelta = event.deltaY * 0.01;
                    break;

                default:
                    // undefined, 0, assume pixels
                    _wheelDelta = event.deltaY * 0.00025;
                    break;
            }

            _this.dispatchEvent(startEvent);
            _this.dispatchEvent(endEvent);
        }

        function contextmenu(event) {
            event.preventDefault();
        }

        this.dispose = function () {
            scope.domElement.removeEventListener('contextmenu', contextmenu);

            scope.domElement.removeEventListener('pointerdown', onPointerDown);
            scope.domElement.removeEventListener('pointercancel', onPointerCancel);
            scope.domElement.removeEventListener('wheel', onMouseWheel);

            scope.domElement.removeEventListener('pointermove', onPointerMove);
            scope.domElement.removeEventListener('pointerup', onPointerUp);
        };

        this.domElement.addEventListener('pointerdown', onPointerDown);
        this.domElement.addEventListener('pointermove', onPointerMove);
        this.domElement.addEventListener('pointerup', onPointerUp);
        this.domElement.addEventListener('pointercancel', onPointerCancel);
        this.domElement.addEventListener('wheel', onMouseWheel, { passive: false });
        this.domElement.addEventListener('contextmenu', contextmenu);

        // force an update at start
        this.handleResize();
        this.update();

        trackBall.update();

        trackBall.noRotate = K3D.parameters.cameraNoRotate;
        trackBall.noZoom = K3D.parameters.cameraNoZoom;
        trackBall.noPan = K3D.parameters.cameraNoPan;
        trackBall.rotateSpeed = K3D.parameters.cameraRotateSpeed * 2;
        trackBall.zoomSpeed = K3D.parameters.cameraZoomSpeed * 2;
        trackBall.panSpeed = K3D.parameters.cameraPanSpeed * 2;
        trackBall.staticMoving = true;
        trackBall.dynamicDampingFactor = 0.1;
        trackBall.initialized = false;

        trackBall.addEventListener('change', (changeEvent) => {
            _this.dispatchEvent(changeEvent);
        });
    };

    THREE.VolumeSidesControls.prototype = Object.create(THREE.EventDispatcher.prototype);
    THREE.VolumeSidesControls.prototype.constructor = THREE.VolumeSidesControls;
};
