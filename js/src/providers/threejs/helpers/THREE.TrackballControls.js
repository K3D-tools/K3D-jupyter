// jshint ignore: start
// jscs:disable

module.exports = function (THREE) {
    THREE.TrackballControls = function (object, domElement) {
        const scope = this;
        const STATE = {
            NONE: -1, ROTATE: 0, ZOOM: 1, PAN: 2, TOUCH_ROTATE: 3, TOUCH_ZOOM_PAN: 4,
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

        // API

        this.enabled = true;

        this.screen = {
            left: 0, top: 0, width: 0, height: 0,
        };

        this.flyMode = false;
        this.rotateSpeed = 1.0;
        this.zoomSpeed = 1.2;
        this.panSpeed = 0.3;

        this.noRotate = false;
        this.noZoom = false;
        this.noPan = false;

        this.staticMoving = false;
        this.dynamicDampingFactor = 0.2;

        this.minDistance = 0;
        this.maxDistance = Infinity;

        this.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };

        // internals

        this.target = new THREE.Vector3();

        const EPS = 0.0000000001;

        const lastPosition = new THREE.Vector3();
        const
            lastUp = new THREE.Vector3();
        let lastZoom = 1;

        let _state = STATE.NONE;
        let _keyState = STATE.NONE;

        let _touchZoomDistanceStart = 0;
        let _touchZoomDistanceEnd = 0;

        let _lastAngle = 0;

        const _eye = new THREE.Vector3();

        const _movePrev = new THREE.Vector2();
        const _moveCurr = new THREE.Vector2();

        const _lastAxis = new THREE.Vector3();

        const _zoomStart = new THREE.Vector2();
        const _zoomEnd = new THREE.Vector2();

        const _panStart = new THREE.Vector2();
        const _panEnd = new THREE.Vector2();

        // for reset

        this.target0 = this.target.clone();
        this.position0 = this.object.position.clone();
        this.up0 = this.object.up.clone();
        this.zoom0 = this.object.zoom;

        // events

        const _changeEvent = { type: 'change' };
        const _startEvent = { type: 'start' };
        const _endEvent = { type: 'end' };

        // methods

        this.handleResize = function () {
            if (this.domElement === currentDocument) {
                this.screen.left = 0;
                this.screen.top = 0;
                this.screen.width = currentWindow.innerWidth;
                this.screen.height = currentWindow.innerHeight;
            } else {
                const box = scope.domElement.getBoundingClientRect();
                // adjustments come from similar code in the jquery offset() function
                const d = scope.domElement.ownerDocument.documentElement;
                this.screen.left = box.left + currentWindow.pageXOffset - d.clientLeft;
                this.screen.top = box.top + currentWindow.pageYOffset - d.clientTop;
                this.screen.width = box.width;
                this.screen.height = box.height;
            }
        };

        const getMouseOnScreen = (function () {
            const vector = new THREE.Vector2();

            return function (pageX, pageY) {
                vector.set(
                    (pageX - scope.screen.left) / scope.screen.width,
                    (pageY - scope.screen.top) / scope.screen.height,
                );

                return vector;
            };
        }());

        const getMouseOnCircle = (function () {
            const vector = new THREE.Vector2();

            return function (pageX, pageY) {
                vector.set(
                    ((pageX - scope.screen.width * 0.5 - scope.screen.left) / (scope.screen.width * 0.5)),
                    ((scope.screen.height + 2 * (scope.screen.top - pageY)) / scope.screen.width),
                );

                return vector;
            };
        }());

        this.rotateCamera = (function () {
            const axis = new THREE.Vector3();
            const quaternion = new THREE.Quaternion();
            const eyeDirection = new THREE.Vector3();
            const objectUpDirection = new THREE.Vector3();
            const objectSidewaysDirection = new THREE.Vector3();
            const moveDirection = new THREE.Vector3();

            return function rotateCamera() {
                moveDirection.set(_moveCurr.x - _movePrev.x, _moveCurr.y - _movePrev.y, 0);
                let angle = moveDirection.length();

                if (angle) {
                    _eye.copy(scope.object.position).sub(scope.target);

                    eyeDirection.copy(_eye).normalize();
                    objectUpDirection.copy(scope.object.up).normalize();
                    objectSidewaysDirection.crossVectors(objectUpDirection, eyeDirection).normalize();

                    objectUpDirection.setLength(_moveCurr.y - _movePrev.y);
                    objectSidewaysDirection.setLength(_moveCurr.x - _movePrev.x);

                    moveDirection.copy(objectUpDirection.add(objectSidewaysDirection));

                    axis.crossVectors(moveDirection, _eye).normalize();

                    angle *= scope.rotateSpeed;
                    quaternion.setFromAxisAngle(axis, angle);

                    _eye.applyQuaternion(quaternion);
                    scope.object.up.applyQuaternion(quaternion);

                    _lastAxis.copy(axis);
                    _lastAngle = angle;
                } else if (!scope.staticMoving && _lastAngle) {
                    _lastAngle *= Math.sqrt(1.0 - scope.dynamicDampingFactor);
                    _eye.copy(scope.object.position).sub(scope.target);
                    quaternion.setFromAxisAngle(_lastAxis, _lastAngle);
                    _eye.applyQuaternion(quaternion);
                    scope.object.up.applyQuaternion(quaternion);
                }

                _movePrev.copy(_moveCurr);
            };
        }());

        this.zoomCamera = function () {
            let factor;

            if (_state === STATE.TOUCH_ZOOM_PAN) {
                factor = _touchZoomDistanceStart / _touchZoomDistanceEnd;
                _touchZoomDistanceStart = _touchZoomDistanceEnd;

                if (scope.object.isPerspectiveCamera) {
                    _eye.multiplyScalar(factor);
                } else if (scope.object.isOrthographicCamera) {
                    scope.object.zoom *= factor;
                    scope.object.updateProjectionMatrix();
                } else {
                    console.warn('THREE.TrackballControls: Unsupported camera type');
                }
            } else {
                if (scope.flyMode || _state === STATE.PAN || _keyState === STATE.PAN) {
                    return;
                }

                factor = 1.0 + (_zoomEnd.y - _zoomStart.y) * scope.zoomSpeed;

                if (factor !== 1.0 && factor > 0.0) {
                    if (scope.object.isPerspectiveCamera) {
                        _eye.multiplyScalar(factor);
                    } else if (scope.object.isOrthographicCamera) {
                        scope.object.zoom /= factor;
                        scope.object.updateProjectionMatrix();
                    } else {
                        console.warn('THREE.TrackballControls: Unsupported camera type');
                    }
                }

                if (scope.staticMoving) {
                    _zoomStart.copy(_zoomEnd);
                } else {
                    _zoomStart.y += (_zoomEnd.y - _zoomStart.y) * scope.dynamicDampingFactor;
                }
            }
        };

        this.panCamera = (function () {
            const mouseChange = new THREE.Vector3();
            const objectUp = new THREE.Vector3();
            const zAxis = new THREE.Vector3();
            const pan = new THREE.Vector3();

            return function panCamera() {
                mouseChange.copy(_panEnd).sub(_panStart);
                mouseChange.z = 0.0;

                if (scope.flyMode || _state === STATE.PAN || _keyState === STATE.PAN) {
                    mouseChange.z = (_zoomEnd.y - _zoomStart.y) * scope.panSpeed * 10.0;

                    if (scope.staticMoving) {
                        _zoomStart.copy(_zoomEnd);
                    } else {
                        _zoomStart.y += (_zoomEnd.y - _zoomStart.y) * scope.dynamicDampingFactor;
                    }
                }

                if (mouseChange.lengthSq()) {
                    if (scope.object.isOrthographicCamera) {
                        const scaleX = (scope.object.right - scope.object.left) / scope.object.zoom
                            / scope.domElement.clientWidth;
                        const scaleY = (scope.object.top - scope.object.bottom) / scope.object.zoom
                            / scope.domElement.clientWidth;

                        mouseChange.x *= scaleX;
                        mouseChange.y *= scaleY;
                    }

                    mouseChange.multiplyScalar(_eye.length() * scope.panSpeed);

                    pan.copy(_eye).cross(scope.object.up).setLength(mouseChange.x);
                    pan.add(objectUp.copy(scope.object.up).setLength(mouseChange.y));
                    pan.add(zAxis.copy(_eye).setLength(mouseChange.z));

                    scope.object.position.add(pan);
                    scope.target.add(pan);

                    if (scope.staticMoving) {
                        _panStart.copy(_panEnd);
                    } else {
                        _panStart.add(mouseChange.subVectors(_panEnd, _panStart)
                            .multiplyScalar(scope.dynamicDampingFactor));
                    }
                }
            };
        }());

        this.checkDistances = function () {
            if (!scope.noZoom || !scope.noPan) {
                if (_eye.lengthSq() > scope.maxDistance * scope.maxDistance) {
                    scope.object.position.addVectors(scope.target, _eye.setLength(scope.maxDistance));
                    _zoomStart.copy(_zoomEnd);
                }

                if (_eye.lengthSq() < scope.minDistance * scope.minDistance) {
                    scope.object.position.addVectors(scope.target, _eye.setLength(scope.minDistance));
                    _zoomStart.copy(_zoomEnd);
                }
            }
        };

        this.update = function () {
            _eye.subVectors(scope.object.position, scope.target);

            if (!scope.noRotate) {
                scope.rotateCamera();
            }

            if (!scope.noZoom) {
                scope.zoomCamera();
            }

            if (!scope.noPan) {
                scope.panCamera();
            }

            scope.object.position.addVectors(scope.target, _eye);

            if (scope.object.isPerspectiveCamera) {
                scope.checkDistances();
                scope.object.lookAt(scope.target);

                if (lastPosition.distanceToSquared(scope.object.position) > EPS
                    || lastUp.distanceToSquared(scope.object.up) > EPS) {
                    lastPosition.copy(scope.object.position);
                    lastUp.copy(scope.object.up);

                    scope.dispatchEvent(_changeEvent);
                }
            } else if (scope.object.isOrthographicCamera) {
                scope.object.lookAt(scope.target);

                if (lastPosition.distanceToSquared(scope.object.position) > EPS || lastZoom !== scope.object.zoom) {
                    scope.dispatchEvent(_changeEvent);
                    lastPosition.copy(scope.object.position);
                    lastZoom = scope.object.zoom;
                }
            } else {
                console.warn('THREE.TrackballControls: Unsupported camera type');
            }
        };

        this.reset = function () {
            _state = STATE.NONE;
            _keyState = STATE.NONE;

            scope.target.copy(scope.target0);
            scope.object.position.copy(scope.position0);
            scope.object.up.copy(scope.up0);
            scope.object.zoom = scope.zoom0;

            scope.object.updateProjectionMatrix();

            _eye.subVectors(scope.object.position, scope.target);

            scope.object.lookAt(scope.target);

            scope.dispatchEvent(_changeEvent);

            lastPosition.copy(scope.object.position);
            lastUp.copy(scope.object.up);
            lastZoom = scope.object.zoom;
        };

        // listeners

        function onPointerDown(event) {
            if (scope.enabled === false) return;

            switch (event.pointerType) {
                case 'mouse':
                case 'pen':
                    onMouseDown(event);
                    break;

                default:
                    break;
                // TODO touch
            }
        }

        function onPointerMove(event) {
            if (scope.enabled === false) return;

            switch (event.pointerType) {
                case 'mouse':
                case 'pen':
                    onMouseMove(event);
                    break;
                default:
                    break;
                // TODO touch
            }
        }

        function onPointerUp(event) {
            if (scope.enabled === false) return;

            switch (event.pointerType) {
                case 'mouse':
                case 'pen':
                    onMouseUp(event);
                    break;
                default:
                    break;
                // TODO touch
            }
        }

        function keydown(event) {
            if (scope.enabled === false) return;

            currentWindow.removeEventListener('keydown', keydown);

            if (_keyState !== STATE.NONE) {
                // nothing
            } else if (event.ctrlKey && !scope.noRotate) {
                _keyState = STATE.ROTATE;
            } else if (event.altKey && !scope.noZoom) {
                _keyState = STATE.ZOOM;
            } else if (event.shiftKey && !scope.noPan) {
                _keyState = STATE.PAN;
            }
        }

        function keyup() {
            if (scope.enabled === false) return;

            _keyState = STATE.NONE;

            currentWindow.addEventListener('keydown', keydown);
        }

        function onMouseDown(event) {
            event.preventDefault();

            if (_state === STATE.NONE) {
                switch (event.button) {
                    case scope.mouseButtons.LEFT:
                        _state = STATE.ROTATE;
                        break;

                    case scope.mouseButtons.MIDDLE:
                        _state = STATE.ZOOM;
                        break;

                    case scope.mouseButtons.RIGHT:
                        _state = STATE.PAN;
                        break;

                    default:
                        _state = STATE.NONE;
                }
            }

            const state = (_keyState !== STATE.NONE) ? _keyState : _state;

            if (state === STATE.ROTATE && !scope.noRotate) {
                _moveCurr.copy(getMouseOnCircle(event.pageX, event.pageY));
                _movePrev.copy(_moveCurr);
            } else if (state === STATE.ZOOM && !scope.noZoom) {
                _zoomStart.copy(getMouseOnScreen(event.pageX, event.pageY));
                _zoomEnd.copy(_zoomStart);
            } else if (state === STATE.PAN && !scope.noPan) {
                _panStart.copy(getMouseOnScreen(event.pageX, event.pageY));
                _panEnd.copy(_panStart);
            }

            currentDocument.addEventListener('pointermove', onPointerMove);
            currentDocument.addEventListener('pointerup', onPointerUp);

            scope.dispatchEvent(_startEvent);
        }

        function onMouseMove(event) {
            if (scope.enabled === false) return;

            event.preventDefault();

            const state = (_keyState !== STATE.NONE) ? _keyState : _state;

            if (state === STATE.ROTATE && !scope.noRotate) {
                _movePrev.copy(_moveCurr);
                _moveCurr.copy(getMouseOnCircle(event.pageX, event.pageY));
            } else if (state === STATE.ZOOM && !scope.noZoom) {
                _zoomEnd.copy(getMouseOnScreen(event.pageX, event.pageY));
            } else if (state === STATE.PAN && !scope.noPan) {
                _panEnd.copy(getMouseOnScreen(event.pageX, event.pageY));
            }
        }

        function onMouseUp(event) {
            if (scope.enabled === false) return;

            event.preventDefault();

            _state = STATE.NONE;

            currentDocument.removeEventListener('pointermove', onPointerMove);
            currentDocument.removeEventListener('pointerup', onPointerUp);

            scope.dispatchEvent(_endEvent);
        }

        function mousewheel(event) {
            if (scope.enabled === false) return;

            if (scope.noZoom === true) return;

            event.preventDefault();

            switch (event.deltaMode) {
                case 2:
                    // Zoom in pages
                    _zoomStart.y -= event.deltaY * 0.025;
                    break;

                case 1:
                    // Zoom in lines
                    _zoomStart.y -= event.deltaY * 0.01;
                    break;

                default:
                    // undefined, 0, assume pixels
                    _zoomStart.y -= event.deltaY * 0.00025;
                    break;
            }

            scope.dispatchEvent(_startEvent);
            scope.dispatchEvent(_endEvent);
        }

        function touchstart(event) {
            if (scope.enabled === false) return;

            event.preventDefault();

            switch (event.touches.length) {
                case 1:
                    _state = STATE.TOUCH_ROTATE;
                    _moveCurr.copy(getMouseOnCircle(event.touches[0].pageX, event.touches[0].pageY));
                    _movePrev.copy(_moveCurr);
                    break;

                default: {
                    // 2 or more
                    _state = STATE.TOUCH_ZOOM_PAN;
                    const dx = event.touches[0].pageX - event.touches[1].pageX;
                    const dy = event.touches[0].pageY - event.touches[1].pageY;
                    _touchZoomDistanceStart = Math.sqrt(dx * dx + dy * dy);
                    _touchZoomDistanceEnd = _touchZoomDistanceStart;

                    const x = (event.touches[0].pageX + event.touches[1].pageX) / 2;
                    const y = (event.touches[0].pageY + event.touches[1].pageY) / 2;
                    _panStart.copy(getMouseOnScreen(x, y));
                    _panEnd.copy(_panStart);
                    break;
                }
            }

            scope.dispatchEvent(_startEvent);
        }

        function touchmove(event) {
            if (scope.enabled === false) return;

            event.preventDefault();

            switch (event.touches.length) {
                case 1:
                    _movePrev.copy(_moveCurr);
                    _moveCurr.copy(getMouseOnCircle(event.touches[0].pageX, event.touches[0].pageY));
                    break;

                default: {
                    // 2 or more
                    const dx = event.touches[0].pageX - event.touches[1].pageX;
                    const dy = event.touches[0].pageY - event.touches[1].pageY;
                    _touchZoomDistanceEnd = Math.sqrt(dx * dx + dy * dy);

                    const x = (event.touches[0].pageX + event.touches[1].pageX) / 2;
                    const y = (event.touches[0].pageY + event.touches[1].pageY) / 2;
                    _panEnd.copy(getMouseOnScreen(x, y));
                    break;
                }
            }
        }

        function touchend(event) {
            if (scope.enabled === false) return;

            switch (event.touches.length) {
                case 0:
                    _state = STATE.NONE;
                    break;

                case 1:
                    _state = STATE.TOUCH_ROTATE;
                    _moveCurr.copy(getMouseOnCircle(event.touches[0].pageX, event.touches[0].pageY));
                    _movePrev.copy(_moveCurr);
                    break;
                default:
                    break;
            }

            scope.dispatchEvent(_endEvent);
        }

        function contextmenu(event) {
            if (scope.enabled === false) return;

            event.preventDefault();
        }

        this.dispose = function () {
            scope.domElement.removeEventListener('contextmenu', contextmenu);

            scope.domElement.removeEventListener('pointerdown', onPointerDown);
            scope.domElement.removeEventListener('wheel', mousewheel);

            scope.domElement.removeEventListener('touchstart', touchstart);
            scope.domElement.removeEventListener('touchend', touchend);
            scope.domElement.removeEventListener('touchmove', touchmove);

            currentDocument.removeEventListener('pointermove', onPointerMove);
            currentDocument.removeEventListener('pointerup', onPointerUp);

            currentWindow.removeEventListener('keydown', keydown);
            currentWindow.removeEventListener('keyup', keyup);
        };

        this.domElement.addEventListener('contextmenu', contextmenu);

        this.domElement.addEventListener('pointerdown', onPointerDown);
        this.domElement.addEventListener('wheel', mousewheel, { passive: false });

        this.domElement.addEventListener('touchstart', touchstart, { passive: false });
        this.domElement.addEventListener('touchend', touchend);
        this.domElement.addEventListener('touchmove', touchmove, { passive: false });

        this.domElement.ownerDocument.addEventListener('pointermove', onPointerMove);
        this.domElement.ownerDocument.addEventListener('pointerup', onPointerUp);

        currentWindow.addEventListener('keydown', keydown);
        currentWindow.addEventListener('keyup', keyup);

        this.handleResize();

        // force an update at start
        this.update();
    };

    THREE.TrackballControls.prototype = Object.create(THREE.EventDispatcher.prototype);
    THREE.TrackballControls.prototype.constructor = THREE.TrackballControls;
};
