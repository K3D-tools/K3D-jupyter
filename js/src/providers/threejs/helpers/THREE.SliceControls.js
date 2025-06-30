const objectsGUIprovider = require('../../../core/lib/objectsGUIprovider');

module.exports = function (THREE) {
    THREE.SliceControls = function (object, domElement, K3D) {
        const scope = this;

        function getSliceJson() {
            const objectId = K3D.parameters.sliceViewerObjectId;
            const json = K3D.getWorld().ObjectsListJson[objectId];

            return json;
        }

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
        let axis;

        const _this = this;
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

        // API

        this.enabled = true;

        this.screen = {
            left: 0, top: 0, width: 0, height: 0,
        };
        const _pointers = [];
        const _pointerPositions = {};

        this._pointerPositions = _pointerPositions;
        this._pointers = _pointers;

        this.minDistance = 0;
        this.maxDistance = Infinity;

        // internals
        this.target = new THREE.Vector3();

        const EPS = 0.0000000001;

        const lastSlices = new THREE.Vector3(-1, -1, -1);

        let _state = STATE.NONE;

        const _mouseCurrent = new THREE.Vector2();
        const _mouseLast = new THREE.Vector2();

        const _zoomPanCurrent = new THREE.Vector3();
        const _zoomPanLast = new THREE.Vector3();

        let _touchZoomDistanceStart = 0;
        let _touchZoomDistanceEnd = 0;
        let _wheelDelta = 0;

        // events

        const changeEvent = {type: 'change'};
        const startEvent = {type: 'start'};
        const endEvent = {type: 'end'};

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

        const getMouseOnScreen = (function () {
            const vector = new THREE.Vector2();

            return function (pageX, pageY) {
                vector.set(
                    (pageX - _this.screen.left) / _this.screen.width,
                    (pageY - _this.screen.top) / _this.screen.height,
                );

                return vector;
            };
        }());

        this.changeSlice = function (json, change) {
            let shape = Array.isArray(json.volume) ? json.volume[0].shape : json.volume.shape;

            json[`slice_${axis}`] += change;
            json[`slice_${axis}`] = Math.max(-1, Math.min(json[`slice_${axis}`], shape[axisToShape[axis]] - 1));

            return change !== 0;
        };

        this.changeLevels = function (json) {
            const change = _mouseCurrent.clone().sub(_mouseLast);
            const mag = (json.color_range[1] - json.color_range[0]) * 0.25;

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

                objectsGUIprovider.changeParameter(K3D, json, 'color_range', json.color_range);

                return true;
            }

            return false;
        };

        this.changeZoom = function (json) {
            const change = _mouseCurrent.clone().sub(_mouseLast);

            if (change.length() > 0) {
                if (Math.abs(change.y) > EPS) {
                    _zoomPanCurrent.z += change.y;
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

            if (change.length() > 0) {
                _zoomPanCurrent.x += change.x * 5;
                _zoomPanCurrent.y += change.y * 5;

                return true;
            }

            return false;
        };

        this.update = function () {
            const json = getSliceJson();
            let ray = new THREE.Vector3();
            let up = new THREE.Vector3();
            const right = new THREE.Vector3();
            let sliceDistance;
            let change = false;

            axis = K3D.parameters.sliceViewerDirection;

            if (typeof (json) === 'undefined') {
                return;
            }

            let shape = Array.isArray(json.volume) ? json.volume[0].shape : json.volume.shape;

            sliceDistance = json[`slice_${axis}`];

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

                if (_touchZoomDistanceEnd !== _touchZoomDistanceStart) {
                    _zoomPanCurrent.z += (_touchZoomDistanceStart - _touchZoomDistanceEnd) * 0.005;
                    change = true;
                }
            }

            if (_state === STATE.TOUCH_LEVELS) {
                change |= _this.changeLevels(json);
            }

            change |= _this.changeSlice(json, Math.sign(_wheelDelta));

            // update camera
            const obj = K3D.getObjectById(json.id);

            if (!obj) {
                return;
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
                sliceDistance = json.volume.shape[0] - 1 - json.slice_z;
            }

            const slicePosition = obj.position.clone().sub(ray.multiplyScalar(0.5)).add(
                ray.clone().multiplyScalar(
                    (2.0 * sliceDistance + 0.1 + 1.0) / (shape[axisToShape[axis]]),
                ),
            );

            slicePosition.add(
                up.clone().multiplyScalar(_zoomPanCurrent.y * 0.5),
            ).add(
                right.clone().multiplyScalar(-_zoomPanCurrent.x * 0.5),
            );

            sliceDistance = (up.length() * 0.5) / Math.tan(
                THREE.Math.degToRad(K3D.parameters.cameraFov / 2.0),
            );
            const camDistance = sliceDistance * (1.0 + _zoomPanCurrent.z * 0.5);

            _this.object.position.copy(slicePosition.clone().add(
                ray.clone().normalize().multiplyScalar(camDistance),
            ));
            _this.object.up.copy(up.clone().normalize());
            _this.target.copy(slicePosition);
            _this.object.lookAt(_this.target);

            // update slice?
            const newSlice = new THREE.Vector3();
            newSlice.set(json.slice_x, json.slice_y, json.slice_z);

            if (lastSlices.distanceToSquared(newSlice) > EPS) {
                const changes = {};

                changes[`slice_${axis}`] = json[`slice_${axis}`];

                objectsGUIprovider.changeParameter(K3D, json, `slice_${axis}`, json[`slice_${axis}`], true);
                objectsGUIprovider.update(K3D, json, K3D.GUI.objects, changes);

                if (K3D.parameters.sliceViewerMaskObjectIds.length > 0) {
                    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
                        ray.clone().normalize(),
                        slicePosition,
                    );

                    // sliceViewerMaskObjectIds
                    K3D.parameters.sliceViewerMaskObjectIds.forEach((objId) => {
                        const sliceJson = K3D.getWorld().ObjectsListJson[objId];
                        const sliceChange = {};

                        if (!sliceJson) {
                            return;
                        }

                        if (sliceJson.opacity === 1.0) {
                            sliceChange.opacity = 0.95;
                            sliceJson.opacity = 0.95;
                        }

                        sliceJson.slice_planes = [
                            [plane.normal.x, plane.normal.y, plane.normal.z,
                                plane.constant - camDistance * 0.005],
                        ];
                        sliceChange.slice_planes = sliceJson.slice_planes;

                        if (sliceJson.visible) {
                            K3D.reload(sliceJson, sliceChange, true);
                        }
                    });
                }
            }

            // discard change
            _wheelDelta = 0;
            lastSlices.copy(newSlice);
            _mouseLast.copy(_mouseCurrent);
            _touchZoomDistanceStart = _touchZoomDistanceEnd;

            if (change) {
                _this.dispatchEvent(changeEvent);
            }
        };

        this.reslice = function () {
            lastSlices.set(-1, -1, -1);
        };

        this.reset = function () {
            _state = STATE.NONE;

            _zoomPanCurrent.set(0.0, 0.0, 0.0);
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

            position.set(event.pageX, event.pageY);
        }

        function getSecondPointerPosition(event) {
            const pointer = (event.pointerId === _pointers[0].pointerId) ? _pointers[1] : _pointers[0];

            return _pointerPositions[pointer.pointerId];
        }

        // listeners

        function onPointerDown(event) {
            if (scope.enabled === false) return;

            if (_pointers.length === 0) {
                scope.domElement.setPointerCapture(event.pointerId);

                scope.domElement.addEventListener('pointermove', onPointerMove);
                scope.domElement.addEventListener('pointerup', onPointerUp);
            }

            addPointer(event);

            if (event.pointerType === 'touch') {
                onTouchStart(event);
            } else {
                onMouseDown(event);
            }
        }

        function onPointerMove(event) {
            if (scope.enabled === false) return;

            if (event.pointerType === 'touch') {
                onTouchMove(event);
            } else {
                onMouseMove(event);
            }
        }

        function onPointerUp(event) {
            if (scope.enabled === false) return;

            if (event.pointerType === 'touch') {
                onTouchEnd();
            } else {
                onMouseUp();
            }
            removePointer(event);

            if (_pointers.length === 0) {
                scope.domElement.releasePointerCapture(event.pointerId);

                scope.domElement.removeEventListener('pointermove', onPointerMove);
                scope.domElement.removeEventListener('pointerup', onPointerUp);
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
                    _mouseCurrent.copy(getMouseOnScreen(_pointers[0].pageX, _pointers[0].pageY));
                    _mouseLast.copy(_mouseCurrent);
                    break;

                default:

                    const position = getSecondPointerPosition(event);

                    const dx = event.pageX - position.x;
                    const dy = event.pageY - position.y;
                    _touchZoomDistanceEnd = _touchZoomDistanceStart = Math.sqrt(dx * dx + dy * dy);

                    const x = (event.pageX + position.x) / 2;
                    const y = (event.pageY + position.y) / 2;
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
            if (_this.enabled === false) {
                return;
            }

            trackPointer(event);

            switch (_pointers.length) {
                case 1:
                    _mouseLast.copy(_mouseCurrent);
                    _mouseCurrent.copy(getMouseOnScreen(event.pageX, event.pageY));
                    break;
                case 2:
                    _mouseLast.copy(_mouseCurrent);

                    const position = getSecondPointerPosition(event);

                    const dx = event.pageX - position.x;
                    const dy = event.pageY - position.y;

                    _touchZoomDistanceEnd = Math.sqrt(dx * dx + dy * dy);

                    const x = (event.pageX + position.x) / 2;
                    const y = (event.pageY + position.y) / 2;
                    _mouseCurrent.copy(getMouseOnScreen(x, y));
                    break;
                default: // 3 or more
                    _mouseLast.copy(_mouseCurrent);
                    _mouseCurrent.copy(getMouseOnScreen(_pointers[0].pageX, _pointers[0].pageY));
                    break;
            }
        }

        function onTouchEnd() {
            if (_this.enabled === false) {
                return;
            }

            _state = STATE.NONE;
            _this.dispatchEvent(endEvent);
        }

        function onMouseDown(event) {
            event.preventDefault();

            if (_this.enabled === false) {
                return;
            }

            if (_state === STATE.NONE) {
                _state = event.button;
            }

            _mouseCurrent.copy(getMouseOnScreen(event.pageX, event.pageY));
            _mouseLast.copy(_mouseCurrent);
            _this.dispatchEvent(startEvent);
        }

        function onMouseMove(event) {
            if (_this.enabled === false) {
                return;
            }
            _mouseCurrent.copy(getMouseOnScreen(event.pageX, event.pageY));
        }

        function onMouseUp() {
            if (_this.enabled === false) {
                return;
            }

            _state = STATE.NONE;
            _this.dispatchEvent(endEvent);
        }

        function onMouseWheel(event) {
            if (_this.enabled === false) {
                return;
            }

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
        this.domElement.addEventListener('pointercancel', onPointerCancel);
        this.domElement.addEventListener('wheel', onMouseWheel, {passive: false});
        this.domElement.addEventListener('contextmenu', contextmenu);

        // force an update at start
        this.handleResize();
        this.update();
    };

    THREE.SliceControls.prototype = Object.create(THREE.EventDispatcher.prototype);
    THREE.SliceControls.prototype.constructor = THREE.SliceControls;
};
