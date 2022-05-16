const fflate = require('fflate');
const msgpack = require('msgpack-lite');

const LilGUI = require('lil-gui').GUI;
const { viewModes } = require('./lib/viewMode');
const _ = require('../lodash');
const { cameraModes } = require('./lib/cameraMode');
const loader = require('./lib/Loader');
const serialize = require('./lib/helpers/serialize');
const screenshot = require('./lib/screenshot');
const snapshot = require('./lib/snapshot');
const resetCameraGUI = require('./lib/resetCamera');
const detachWindowGUI = require('./lib/detachWindow');
const fullscreen = require('./lib/fullscreen');
const { viewModeGUI } = require('./lib/viewMode');
const { cameraModeGUI } = require('./lib/cameraMode');
const manipulate = require('./lib/manipulate');
const { getColorLegend } = require('./lib/colorMapLegend');
const objectsGUIProvider = require('./lib/objectsGUIprovider');
const clippingPlanesGUIProvider = require('./lib/clippingPlanesGUIProvider');
const timeSeries = require('./lib/timeSeries');
const { base64ToArrayBuffer } = require('./lib/helpers/buffer');

const MsgpackCodec = msgpack.createCodec({ preset: true });

window.Float16Array = require('./lib/helpers/float16Array');

MsgpackCodec.addExtPacker(0x20, Float16Array, (val) => val);
MsgpackCodec.addExtUnpacker(0x20, (val) => Float16Array(val.buffer));

/**
 * @constructor Core
 * @memberof K3D
 * @param {Object} provider provider that will be used by current instance
 * @param {Node} targetDOMNode a handler for a target DOM canvas node
 * @param {Object} parameters of plot
 */
function K3D(provider, targetDOMNode, parameters) {
    /**
     * Current K3D instance world
     * @private
     * @type {Object}
     * @name world
     * @memberof K3D.Core
     * @inner
     * @property {Node} targetDOMNode a handler for a target DOM canvas node
     * @property {Object} di an key-value hash of any external dependencies required
     */
    const self = this;
    let fpsMeter = null;
    let objectIndex = 1;
    const currentWindow = targetDOMNode.ownerDocument.defaultView
        || targetDOMNode.ownerDocument.parentWindow;
    const world = {
        ObjectsListJson: {},
        ObjectsById: {},
        chunkList: {},
        targetDOMNode,
        overlayDOMNode: null,
    };
    let listeners = {};
    let listenersIndex = 0;
    const GUI = {
        controls: null,
        objects: null,
    };

    let guiContainer;

    require('../k3d.css');

    function dispatch(eventName, data) {
        if (!listeners[eventName]) {
            return false;
        }

        Object.keys(listeners[eventName]).forEach((key) => {
            listeners[eventName][key](data);
        });

        return true;
    }

    function changeParameters(key, value) {
        dispatch(self.events.PARAMETERS_CHANGE, {
            key,
            value,
        });
    }

    function initializeGUI() {
        self.gui = new LilGUI({
            width: 220, autoPlace: false, title: 'K3D panel'
        });

        guiContainer.appendChild(self.gui.domElement);

        GUI.controls = self.gui.addFolder('Controls').close();
        GUI.objects = self.gui.addFolder('Objects').close();
        GUI.info = self.gui.addFolder('Info').close();

        screenshot.screenshotGUI(GUI.controls, self);
        snapshot.snapshotGUI(GUI.controls, self);
        resetCameraGUI(GUI.controls, self);

        if (currentWindow === window) {
            detachWindowGUI(GUI.controls, self);

            if (fullscreen.isAvailable()) {
                fullscreen.initialize(world.targetDOMNode, GUI.controls, currentWindow);
            }
        }

        GUI.controls.add(self.parameters, 'cameraAutoFit').onChange(changeParameters.bind(self,
            'camera_auto_fit'));
        GUI.controls.add(self.parameters, 'gridAutoFit').onChange((value) => {
            self.setGridAutoFit(value);
            changeParameters.call(self, 'grid_auto_fit', value);
        });
        GUI.controls.add(self.parameters, 'gridVisible').onChange((value) => {
            self.setGridVisible(value);
            changeParameters.call(self, 'grid_visible', value);
        });
        GUI.controls.add(self.parameters, 'fpsMeter').onChange((value) => {
            self.setFpsMeter(value);
            changeParameters.call(self, 'fps_meter', value);
        });

        viewModeGUI(GUI.controls, self);
        cameraModeGUI(GUI.controls, self);
        manipulate.manipulateGUI(GUI.controls, self, changeParameters);

        GUI.controls.add(self.parameters, 'cameraFov').step(0.1).min(1.0).max(179)
            .name('FOV')
            .onChange((value) => {
                self.setCameraFOV(value);
                changeParameters.call(self, 'camera_fov', value);
            });
        GUI.controls.add(self.parameters, 'voxelPaintColor').step(1).min(0).max(255)
            .name('voxelColor')
            .onChange(
                changeParameters.bind(self, 'voxel_paint_color'),
            );
        GUI.controls.add(self.parameters, 'lighting').step(0.01).min(0).max(4)
            .name('lighting')
            .onChange((value) => {
                self.setDirectionalLightingIntensity(value);
                changeParameters.call(self, 'lighting', value);
            });

        timeSeries.timeSeriesGUI(GUI.controls, self, changeParameters);

        GUI.clippingPlanes = GUI.controls.addFolder('Clipping planes').close();

        // Info box
        GUI.info.add(self.parameters, 'guiVersion').name('Js version:');
        GUI.info.controllers[0].$input.readOnly = true;

        if (self.parameters.backendVersion) {
            GUI.info.add({
                version: self.parameters.backendVersion,
            }, 'version').name('Python version:');
            GUI.info.controllers[1].$input.readOnly = true;
        }

        Object.keys(world.ObjectsListJson).forEach((id) => {
            objectsGUIProvider.update(self, world.ObjectsListJson[id], GUI.objects, null);
        });
    }

    function removeObjectFromScene(id) {
        let object = self.Provider.Helpers.getObjectById(world, id);

        if (object) {
            world.K3DObjects.remove(object);
            delete world.ObjectsById[id];

            if (object.onRemove) {
                object.onRemove();
            }

            if (object.geometry) {
                object.geometry.dispose();
                object.geometry = undefined;
            }

            if (object.material && object.material.map) {
                object.material.map.dispose();
                object.material.map = undefined;
            }

            if (object.material) {
                object.material.dispose();
                object.material = undefined;
            }

            if (object.mesh) {
                object.mesh.dispose();
                object.mesh = undefined;
            }

            object = undefined;
        }
    }

    if (!(this instanceof (K3D))) {
        return new K3D(provider, targetDOMNode, parameters);
    }

    if (typeof (provider) !== 'object') {
        throw new Error('Provider should be an object (a key-value map following convention)');
    }

    this.refreshAfterObjectsChange = function (isUpdate, force) {
        if (self.parameters.autoRendering || force) {
            if (!isUpdate) {
                self.getWorld().setCameraToFitScene();
            }

            if (GUI.controls) {
                timeSeries.refreshTimeScale(self, GUI);
            }

            if (!isUpdate) {
                return self.rebuildSceneData().then(self.render.bind(null, true));
            }
            return self.render(true);
        }

        return false;
    };

    this.render = function (force) {
        world.render(force);
    };

    this.resizeHelper = function () {
        if (!self.disabling) {
            if (self.gui) {
                self.gui.domElement.parentNode.style['max-height'] = `${world.targetDOMNode.offsetHeight}px`;
            }

            self.Provider.Helpers.resizeListener(world);
            dispatch(self.events.RESIZED);
            self.render();
        }
    };

    world.overlayDOMNode = currentWindow.document.createElement('div');
    world.overlayDOMNode.style.cssText = [
        'position: absolute',
        'width: 100%',
        'height: 100%',
        'top: 0',
        'right: 0',
        'pointer-events: none',
        'overflow: hidden',
    ].join(';');

    this.GUI = GUI;
    this.parameters = _.assignWith({
            viewMode: viewModes.view,
            cameraMode: cameraModes.trackball,
            manipulateMode: manipulate.manipulateModes.translate,
            voxelPaintColor: 0,
            snapshotIncludeJs: true,
            menuVisibility: true,
            cameraAutoFit: true,
            gridAutoFit: true,
            gridVisible: true,
            grid: [-1, -1, -1, 1, 1, 1],
            gridColor: 0xe6e6e6,
            labelColor: 0x444444,
            antialias: 1,
            logarithmicDepthBuffer: true,
            screenshotScale: 5.0,
            renderingSteps: 1,
            clearColor: 0xffffff,
            clippingPlanes: [],
            fpsMeter: false,
            lighting: 1.5,
            time: 0.0,
            colorbarObjectId: -1,
            colorbarScientific: false,
            fps: 25.0,
            axes: ['x', 'y', 'z'],
            cameraNoRotate: false,
            cameraNoZoom: false,
            cameraNoPan: false,
            cameraRotateSpeed: 1.0,
            cameraZoomSpeed: 1.2,
            cameraPanSpeed: 0.3,
            cameraDampingFactor: 0.0,
            name: null,
            cameraFov: 60.0,
            cameraAnimation: {},
            autoRendering: true,
            axesHelper: 1.0,
            axesHelperColors: [0xff0000, 0x00ff00, 0x0000ff],
            snapshotType: 'full',
            customData: null,
            guiVersion: require('../../package.json').version,
        },
        parameters || {},
        (objValue, srcValue) => (typeof (srcValue) === 'undefined' ? objValue : srcValue));

    this.autoRendering = false;

    this.startAutoPlay = function () {
        timeSeries.startAutoPlay(self, changeParameters);
    };

    this.stopAutoPlay = function () {
        timeSeries.stopAutoPlay(self);
    };

    this.setFps = function (fps) {
        self.parameters.fps = fps;

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'fps') {
                    controller.updateDisplay();
                }
            });
        }
    };

    this.setFpsMeter = function (state) {
        let Stats;

        if (state) {
            if (fpsMeter) {
                return;
            }

            Stats = require('stats.js');
            fpsMeter = new Stats();

            fpsMeter.dom.style.position = 'absolute';
            world.targetDOMNode.appendChild(fpsMeter.dom);
            requestAnimationFrame(function loop() {
                if (fpsMeter) {
                    fpsMeter.update();
                    requestAnimationFrame(loop);
                }
            });
        } else if (fpsMeter) {
            fpsMeter.domElement.remove();
            fpsMeter = null;
        }

        self.parameters.fpsMeter = state;

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'fpsMeter') {
                    controller.updateDisplay();
                }
            });
        }
    };

    /**
     * Set autoRendering state
     * @memberof K3D.Core
     */
    this.refreshAutoRenderingState = function () {
        let handlersCount = 0;
        Object.keys(self.frameUpdateHandlers).forEach((when) => {
            handlersCount += self.frameUpdateHandlers[when].length;
        });

        self.autoRendering = handlersCount > 0;
    };

    this.dispatch = dispatch;

    /**
     * Stores give provider
     * @memberof K3D.Core
     * @type {Object}
     */
    this.Provider = provider;

    this.setFullscreen = function (state) {
        if (state) {
            fullscreen.screenfull.request(world.targetDOMNode);
        } else {
            fullscreen.screenfull.exit();
        }
    };

    this.getFullscreen = function () {
        return fullscreen.screenfull.isFullscreen;
    }

    this.setDirectionalLightingIntensity = function (value) {
        self.parameters.lighting = Math.min(Math.max(value, 0.0), 4.0);
        self.getWorld().recalculateLights(self.parameters.lighting);
        self.render();

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'lighting') {
                    controller.updateDisplay();
                }
            });
        }
    };

    /**
     * Set view mode of K3D
     * @memberof K3D.Core
     * @param {String} mode
     */
    this.setViewMode = function (mode) {
        self.parameters.viewMode = mode;

        if (dispatch(self.events.VIEW_MODE_CHANGE, mode)) {
            self.render();
        }

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'viewMode') {
                    controller.updateDisplay();
                }
            });

            manipulate.refreshManipulateGUI(self, GUI);
        }

        world.targetDOMNode.style.cursor = 'auto';
    };

    /**
     * Set camera mode of K3D
     * @memberof K3D.Core
     * @param {String} mode
     */
    this.setCameraMode = function (mode) {
        if (typeof (_.invert(cameraModes)[mode]) === 'undefined') {
            mode = cameraModes.trackball;
        }

        self.parameters.cameraMode = mode;
        self.getWorld().changeControls();
        self.getWorld().setCameraToFitScene(true);

        dispatch(self.events.CAMERA_MODE_CHANGE, mode);
        self.render();

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'cameraMode') {
                    controller.updateDisplay();
                }
            });
        }
    };
    /**
     * Set manipulate mode of K3D
     * @memberof K3D.Core
     * @param {String} mode
     */
    this.setManipulateMode = function (mode) {
        self.parameters.manipulateMode = mode;

        if (dispatch(self.events.MANIPULATE_MODE_CHANGE, mode)) {
            self.render();
        }

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'manipulateMode') {
                    controller.updateDisplay();
                }
            });
        }
    };

    /**
     * Set auto rendering of K3D
     * @memberof K3D.Core
     * @param {Bool} flag
     */
    this.setAutoRendering = function (flag) {
        self.parameters.autoRendering = flag;
    };

    /**
     * Set menu visibility of K3D
     * @memberof K3D.Core
     * @param {Boolean} mode
     */
    this.setMenuVisibility = function (mode) {
        self.parameters.menuVisibility = mode;

        if (mode) {
            if (!self.gui) {
                initializeGUI();
            }
        } else if (self.gui) {
            self.gui_map = {};
            self.gui_groups = {};
            self.gui_counts = {};
            self.gui.destroy();
            self.gui.domElement.remove();

            self.gui = null;
        }
    };

    this.setClippingPlanes = function (newPlanes) {
        const planes = _.cloneDeep(newPlanes);
        self.parameters.clippingPlanes.length = 0;

        planes.forEach((p) => {
            self.parameters.clippingPlanes.push(p);
        });

        if (GUI.clippingPlanes) {
            clippingPlanesGUIProvider(self, GUI.clippingPlanes);
        }

        self.render();
    };

    this.setColorbarScientific = function (flag) {
        self.parameters.colorbarScientific = flag;
        self.render();
    };

    this.setColorMapLegend = function (v) {
        const newValue = v.id || v;

        if (self.parameters.colorbarObjectId !== newValue) {
            self.parameters.colorbarObjectId = newValue;
            changeParameters('colorbar_object_id', self.parameters.colorbarObjectId);

            Object.keys(world.ObjectsListJson).forEach((id) => {
                if (world.ObjectsListJson[id].colorLegend) {
                    world.ObjectsListJson[id].colorLegend = false;
                }
            });

            if (newValue > 0 && typeof (world.ObjectsListJson[newValue]) !== 'undefined') {
                world.ObjectsListJson[newValue].colorLegend = true;
            }

            if (GUI.objects) {
                Object.keys(GUI.objects.folders).forEach((k) => {
                    GUI.objects.folders[k].controllers.forEach((controller) => {
                        if (controller.property === 'colorLegend') {
                            controller.updateDisplay();
                        }
                    });
                });
            }
        }

        getColorLegend(self, world.ObjectsListJson[self.parameters.colorbarObjectId] || v);
    };

    /**
     * Set camera auto fit mode of K3D
     * @memberof K3D.Core
     * @param {String} state
     */
    this.setCameraAutoFit = function (state) {
        self.parameters.cameraAutoFit = state;

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'cameraAutoFit') {
                    controller.updateDisplay();
                }
            });
        }

        if (state) {
            self.getWorld().setCameraToFitScene();
        }
    };

    /**
     * Set rendering steps of K3D
     * @memberof K3D.Core
     * @param {String} steps
     */
    this.setRenderingSteps = function (steps) {
        self.parameters.renderingSteps = steps;
    };

    /**
     * Set axes labels of plot
     * @memberof K3D.Core
     * @param {String} axesLabel
     */
    this.setAxes = function (axesLabel) {
        self.parameters.axes = axesLabel;

        self.rebuildSceneData(true).then(() => {
            self.render();
        });
    };

    /**
     * Set name of plot
     * @memberof K3D.Core
     * @param {String} name
     */
    this.setName = function (name) {
        self.parameters.name = name;
    };

    /**
     * Set axes helper of plot
     * @memberof K3D.Core
     * @param {Number} size
     */
    this.setAxesHelper = function (size) {
        self.parameters.axesHelper = size;

        self.rebuildSceneData(true).then(() => {
            self.render();
        });
    };

    /**
     * Set axes helper of plot
     * @memberof K3D.Core
     * @param {Number} size
     */
    this.setAxesHelperColors = function (colors) {
        self.parameters.axesHelperColors = colors;

        self.rebuildSceneData(true).then(() => {
            self.render();
        });
    };

    /**
     * Set grid auto fit mode of K3D
     * @memberof K3D.Core
     * @param {String} state
     */
    this.setGridAutoFit = function (state) {
        self.parameters.gridAutoFit = state;

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'gridAutoFit') {
                    controller.updateDisplay();
                }
            });
        }
    };

    /**
     * Set camera lock
     * @memberof K3D.Core
     * @param {Boolean} cameraNoRotate
     * @param {Boolean} cameraNoZoom
     * @param {Boolean} cameraNoPan
     */
    this.setCameraLock = function (cameraNoRotate, cameraNoZoom, cameraNoPan) {
        self.parameters.cameraNoRotate = cameraNoRotate;
        self.parameters.cameraNoZoom = cameraNoZoom;
        self.parameters.cameraNoPan = cameraNoPan;

        world.controls.noRotate = cameraNoRotate;
        world.controls.noZoom = cameraNoZoom;
        world.controls.noPan = cameraNoPan;
    };

    /**
     * Set camera speed
     * @memberof K3D.Core
     * @param {Number} rotateSpeed
     * @param {Number} zoomSpeed
     * @param {Number} panSpeed
     */
    this.setCameraSpeeds = function (rotateSpeed, zoomSpeed, panSpeed) {
        self.parameters.cameraRotateSpeed = rotateSpeed;
        self.parameters.cameraZoomSpeed = zoomSpeed;
        self.parameters.cameraPanSpeed = panSpeed;

        world.controls.rotateSpeed = rotateSpeed;
        world.controls.zoomSpeed = zoomSpeed;
        world.controls.panSpeed = panSpeed;
    };

    /**
     * Set camera field of view
     * @memberof K3D.Core
     * @param {Number} angle
     */
    this.setCameraFOV = function (angle) {
        self.parameters.cameraFov = angle;
        world.setupCamera(null, angle);

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'cameraFov') {
                    controller.updateDisplay();
                }
            });
        }

        self.rebuildSceneData(false).then(() => {
            self.render();
        });
    };

    /**
     * Set camera damping factor
     * @memberof K3D.Core
     * @param {Float} factor
     */
    this.setCameraDampingFactor = function (factor) {
        self.parameters.cameraDampingFactor = factor;

        self.getWorld().changeControls(true);

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'damping_factor') {
                    controller.updateDisplay();
                }
            });
        }
    };

    /**
     * Set grid auto fit mode of K3D
     * @memberof K3D.Core
     * @param {String} state
     */
    this.setGridVisible = function (state) {
        self.parameters.gridVisible = state;

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'gridVisible') {
                    controller.updateDisplay();
                }
            });
        }

        self.refreshGrid();
        self.render();
    };

    /**
     * Set grid color of K3D
     * @memberof K3D.Core
     * @param {Number} color
     */
    this.setGridColor = function (color) {
        self.parameters.gridColor = color;
        self.rebuildSceneData().then(() => {
            self.render();
        });
    };

    /**
     * Set labels color of K3D
     * @memberof K3D.Core
     * @param {Number} color
     */
    this.setLabelColor = function (color) {
        self.parameters.labelColor = color;
        self.rebuildSceneData(true).then(() => {
            self.render();
        });
    };

    /**
     * Set screenshot scale for K3D
     * @memberof K3D.Core
     * @param {Number} scale
     */
    this.setScreenshotScale = function (scale) {
        self.parameters.screenshotScale = scale;
    };

    /**
     * Set snapshot include param for K3D
     * @memberof K3D.Core
     * @param {String} state
     */
    this.setSnapshotType = function (state) {
        self.parameters.snapshotType = state;
    };

    /**
     * Set grid of K3D
     * @memberof K3D.Core
     * @param {Array} vectors
     */
    this.setGrid = function (vectors) {
        self.parameters.grid = vectors;

        self.rebuildSceneData(true).then(() => {
            self.refreshGrid();
            self.render();
        });
    };

    /**
     * Set camera of K3D
     * @memberof K3D.Core
     * @param {Object} camera
     */
    this.setCamera = function (camera) {
        if (camera.length > 0) {
            world.setupCamera(camera);
        }
    };

    /**
     * Set camera animation of K3D
     * @memberof K3D.Core
     * @param {Object} config
     */
    this.setCameraAnimation = function (config) {
        self.parameters.cameraAnimation = config;

        if (GUI.controls) {
            timeSeries.refreshTimeScale(self, GUI);
        }
    };

    /**
     * Reset camera of K3D
     * @memberof K3D.Core
     */
    this.resetCamera = function (factor) {
        world.setCameraToFitScene(true, factor);
        world.render();
    };

    /**
     * Set voxelPaintColor of K3D
     * @memberof K3D.Core
     * @param {Number} color
     */
    this.setVoxelPaint = function (color) {
        self.parameters.voxelPaintColor = color;

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'voxelPaintColor') {
                    controller.updateDisplay();
                }
            });
        }
    };

    /**
     * Set clear color in renderer
     * @memberof K3D.Core
     * @param color {Number}
     */
    this.setClearColor = function (color) {
        self.parameters.clearColor = color;

        if (color >= 0) {
            const newColor = parseInt(color, 10) + 0x1000000;
            world.targetDOMNode.style.backgroundColor = `#${newColor.toString(16).substr(1)}`;
        } else {
            world.targetDOMNode.style.backgroundColor = '#fff';
        }
    };

    this.on = function (eventName, listener) {
        listeners[eventName] = listeners[eventName] || {};
        listeners[eventName][listenersIndex] = listener;

        listenersIndex += 1;
        return listenersIndex - 1;
    };

    this.off = function (eventName, id) {
        listeners[eventName] = listeners[eventName] || {};
        delete listeners[eventName][id];
    };

    /**
     * Get access to Scene in current world
     * @memberof K3D.Core
     * @returns {Object|undefined} - should return the "scene" if provider uses such a thing
     */
    this.getScene = function () {
        return world.scene;
    };

    /**
     * Add or update object to K3D objects in current world
     * @memberof K3D.Core
     * @param {Object} object
     * @param {Object} K3DObject
     */
    this.addOrUpdateObject = function (object, K3DObject) {
        try {
            removeObjectFromScene(object.id);
        } catch (e) {
            // nothing
        }

        if (object.visible !== false) {
            world.K3DObjects.add(K3DObject);
        }

        objectIndex += 1;

        return objectIndex;
    };

    /**
     * Get Object instance by id
     * @memberof K3D.Core
     * @param {Number} id
     */
    this.getObjectById = function (id) {
        return self.Provider.Helpers.getObjectById(world, id);
    };

    /**
     * Set ChunkList
     * @memberof K3D.Core
     * @param {Object} json
     */
    this.setChunkList = function (json) {
        world.chunkList = json;
    };

    /**
     * Remove object from current world
     * @memberof K3D.Core
     * @param {String} id
     */
    this.removeObject = function (id) {
        removeObjectFromScene(id);
        delete world.ObjectsListJson[id];

        if (id === self.parameters.colorbarObjectId) {
            self.setColorMapLegend(-1);
        }

        dispatch(self.events.OBJECT_REMOVED, id);
        self.refreshAfterObjectsChange(false);

        return Promise.resolve(true);
    };

    /**
     * Set time of the scene. Used by TimeSeries properties
     * @memberof K3D.Core
     * @public
     * @param {Number} time time in seconds
     */
    this.setTime = function (time) {
        const timeSeriesInfo = timeSeries.getObjectsWithTimeSeriesAndMinMax(self);

        self.parameters.time = Math.min(Math.max(time, timeSeriesInfo.min), timeSeriesInfo.max);

        const promises = timeSeriesInfo.objects.reduce((previousValue, obj) => {
            previousValue.push(self.reload(obj, null, true));

            return previousValue;
        }, []);

        if (Object.keys(self.parameters.cameraAnimation).length > 0) {
            const json = {
                camera: self.parameters.cameraAnimation,
            };
            json.camera.timeSeries = true;

            const newCamera = timeSeries.interpolateTimeSeries(json, time);

            world.setupCamera(newCamera.json.camera);
        }

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'time') {
                    controller.updateDisplay();
                }
            });
        }

        return Promise.all(promises).then(() => {
            self.refreshAfterObjectsChange(true);
        });
    };

    /**
     * A convenient shortcut for doing K3D.Loader(K3DInstance, json);
     * @memberof K3D.Core
     * @public
     * @param {Object} json K3D-JSON object
     * @throws {Error} If Loader fails
     */
    this.load = function (json) {
        return loader(self, json).then((objects) => {
            objects.forEach((object) => {
                objectsGUIProvider.update(self, object.json, GUI.objects, null);

                world.ObjectsListJson[object.json.id] = object.json;
                world.ObjectsById[object.json.id] = object.obj;

                if ((self.parameters.colorbarObjectId === -1
                        && object.json.color_range
                        && object.json.color_range.length === 2)
                    || self.parameters.colorbarObjectId === object.json.id) { // auto
                    self.setColorMapLegend(object.json);
                }
            });

            dispatch(self.events.OBJECT_LOADED);
            self.refreshAfterObjectsChange(false);

            return objects;
        });
    };

    /**
     * Reload object in current world
     * @memberof K3D.Core
     * @param {Object} json
     * @param {Object} changes
     * @param {Bool} timeSeriesReload
     */
    this.reload = function (json, changes, timeSeriesReload) {
        if (json.visible === false) {
            if (timeSeriesReload !== true) {
                self.refreshAfterObjectsChange(true);
                objectsGUIProvider.update(self, json, GUI.objects, changes);
            }

            try {
                removeObjectFromScene(json.id);
            } catch (e) {
                // nothing
            }

            return Promise.resolve(true);
        }

        const data = { objects: [json] };

        if (changes !== null) {
            data.changes = [changes];
        }

        return loader(self, data).then((objects) => {
            objects.forEach((object) => {
                if (timeSeriesReload !== true) {
                    objectsGUIProvider.update(self, object.json, GUI.objects, changes);
                }

                world.ObjectsListJson[object.json.id] = object.json;
                world.ObjectsById[object.json.id] = object.obj;

                if ((self.parameters.colorbarObjectId === -1
                        && object.json.color_range
                        && object.json.color_range.length === 2)
                    || self.parameters.colorbarObjectId === object.json.id) { // auto
                    self.setColorMapLegend(object.json);
                }
            });

            dispatch(self.events.OBJECT_LOADED);

            if (timeSeriesReload !== true) {
                self.refreshAfterObjectsChange(true);
            }

            return objects;
        });
    };

    /**
     * Get access to the whole World
     * @memberof K3D.Core
     * @returns {Object|undefined} - should return the "world" if provider uses such a thing
     */
    this.getWorld = function () {
        return world;
    };

    /**
     * Get Screenshot
     * @memberof K3D.Core
     * @param {Number} scale
     * @param {boolean} onlyCanvas
     * @returns {Canvas|undefined}
     */
    this.getScreenshot = function (scale, onlyCanvas) {
        return screenshot.getScreenshot(this, scale, onlyCanvas);
    };

    /**
     * Get HTML snapshot
     * @memberof K3D.Core
     * @returns {String|undefined}
     */
    this.getHTMLSnapshot = function (compressionLevel) {
        return snapshot.getHTMLSnapshot(this, compressionLevel);
    };

    /**
     * Get snapshot
     * @memberof K3D.Core
     * @returns {String|undefined}
     */
    this.getSnapshot = function (compressionLevel) {
        const chunkList = Object.keys(world.chunkList).reduce((p, k) => {
            const attributes = world.chunkList[k].attributes;

            p[k] = Object.keys(attributes).reduce((prev, key) => {
                prev[key] = serialize.serialize(attributes[key]);

                return prev;
            }, {});

            return p;
        }, {});

        const serializedObjects = _.values(world.ObjectsListJson).map((o) => Object.keys(o)
            .reduce((p, k) => {
                p[k] = serialize.serialize(o[k]);

                return p;
            }, {}));

        const plot = _.cloneDeep(self.parameters);
        plot.camera = self.getWorld().controls.getCameraArray();

        return fflate.zlibSync(
            msgpack.encode(
                {
                    objects: serializedObjects,
                    chunkList,
                    plot,
                },
                { codec: MsgpackCodec },
            ),
            { level: compressionLevel },
        );
    };

    /**
     * Set snapshot
     * @memberof K3D.Core
     */
    this.setSnapshot = function (data) {
        if (typeof (data) === 'string') {
            data = fflate.unzlibSync(new Uint8Array(base64ToArrayBuffer(data)));
        }

        if (data instanceof Uint8Array) {
            data = msgpack.decode(data, { codec: MsgpackCodec });
        }

        Object.keys(data.chunkList).forEach((k) => {
            const chunk = data.chunkList[k];
            world.chunkList[chunk.id] = {
                attributes: Object.keys(chunk).reduce((prev, p) => {
                    prev[p] = serialize.deserialize(chunk[p]);
                    return prev;
                }, {}),
            };
        });

        data.objects.forEach((o) => {
            Object.keys(o).forEach((k) => {
                o[k] = serialize.deserialize(o[k]);
            });
        });

        return self.load({ objects: data.objects }).then(() => self.refreshAfterObjectsChange(false,
            true));
    };

    /**
     * Extract snapshot
     * @memberof K3D.Core
     * @param {String} data
     * @returns {Object|undefined}
     */
    this.extractSnapshot = function (data) {
        return data.match(/var data = '(.+)';/mi);
    };

    /**
     * Destroy logic for current instance. Will remove listeners (browser and owned)
     * @memberof K3D.Core
     */
    this.disable = function () {
        this.disabling = true;
        this.frameUpdateHandlers.before = [];
        this.frameUpdateHandlers.after = [];
        if (this.gui) {
            this.gui.destroy();
        }
        this.autoRendering = false;

        world.K3DObjects.children.forEach((obj) => {
            removeObjectFromScene(obj.K3DIdentifier);
            delete world.ObjectsListJson[obj.K3DIdentifier];
        });
        world.cleanup();

        if (fpsMeter) {
            fpsMeter.domElement.remove();
        }

        listeners = {};
        currentWindow.removeEventListener('resize', this.resizeHelper);
        world.renderer.removeContextLossListener();
        world.renderer.forceContextLoss();
    };

    world.targetDOMNode.appendChild(world.overlayDOMNode);

    this.Provider.Initializers.Renderer.call(world, this);
    this.Provider.Initializers.Setup.call(world, this);
    this.Provider.Initializers.Camera.call(world, this);
    this.Provider.Initializers.Canvas.call(world, this);
    this.Provider.Initializers.Scene.call(world, this);
    this.Provider.Initializers.Manipulate.call(world, this);

    currentWindow.addEventListener('resize', this.resizeHelper, false);

    // load toolbars
    guiContainer = currentWindow.document.createElement('div');
    guiContainer.className = 'dg';
    guiContainer.style.cssText = [
        'position: absolute',
        'top: 0',
        'color: black',
        'right: 0',
        'z-index: 16777271',
        `max-height: ${targetDOMNode.clientHeight}px`,
    ].join(';');
    world.targetDOMNode.appendChild(guiContainer);

    if (self.parameters.menuVisibility) {
        initializeGUI();
    }

    this.resizeHelper();

    self.setScreenshotScale(self.parameters.screenshotScale);
    self.setClearColor(self.parameters.clearColor);
    self.setMenuVisibility(self.parameters.menuVisibility);
    self.setTime(self.parameters.time);
    self.setGridAutoFit(self.parameters.gridAutoFit);
    self.setGridVisible(self.parameters.gridVisible);
    self.setCameraAutoFit(self.parameters.cameraAutoFit);
    self.setCameraDampingFactor(self.parameters.cameraDampingFactor);
    self.setClippingPlanes(self.parameters.clippingPlanes);
    self.setDirectionalLightingIntensity(self.parameters.lighting);
    self.setColorMapLegend(self.parameters.colorbarObjectId);
    self.setColorbarScientific(self.parameters.colorbarScientific);
    self.setAutoRendering(self.parameters.autoRendering);
    self.setCameraLock(
        self.parameters.cameraNoRotate,
        self.parameters.cameraNoZoom,
        self.parameters.cameraNoPan,
    );
    self.setCameraSpeeds(
        self.parameters.cameraRotateSpeed,
        self.parameters.cameraZoomSpeed,
        self.parameters.cameraPanSpeed,
    );
    self.setCameraFOV(self.parameters.cameraFov);
    self.setFps(self.parameters.fps);
    self.setViewMode(self.parameters.viewMode);
    self.setFpsMeter(self.parameters.fpsMeter);

    self.MsgpackCodec = MsgpackCodec;
    self.msgpack = msgpack;
    self.serialize = serialize;

    self.render();

    world.targetDOMNode.className += ' k3d-target';
}

function isSupportedUpdateListener(when) {
    return (when in {
        before: !0,
        after: !0,
    });
}

/**
 * Hash with before and after render listeners
 * @memberof K3D.Core
 * @public
 * @property {Array.<Function>} before Before render listeners
 * @property {Array.<Function>} after After render listeners
 * @type {Object}
 */
K3D.prototype.frameUpdateHandlers = {
    before: [],
    after: [],
};

K3D.prototype.events = {
    VIEW_MODE_CHANGE: 'viewModeChange',
    CAMERA_MODE_CHANGE: 'cameraModeChange',
    MANIPULATE_MODE_CHANGE: 'manipulateModeChange',
    RENDERED: 'rendered',
    BEFORE_RENDER: 'before_render',
    RESIZED: 'resized',
    CAMERA_CHANGE: 'cameraChange',
    OBJECT_LOADED: 'objectLoaded',
    OBJECT_REMOVED: 'objectRemoved',
    OBJECT_CHANGE: 'objectChange',
    OBJECT_HOVERED: 'objectHovered',
    OBJECT_CLICKED: 'objectClicked',
    PARAMETERS_CHANGE: 'parametersChange',
    VOXELS_CALLBACK: 'voxelsCallback',
    MOUSE_MOVE: 'mouseMove',
    MOUSE_CLICK: 'mouseClick',
};

/**
 * Attach a listener to current instance
 * @memberof K3D.Core
 * @public
 * @param {String} when=before  When to call the listener (after or before)
 * @param {Function} listener   The listener to be called when before- after- render
 * @param {Bool} callOnce       Info if this is a single-call listener
 */
K3D.prototype.addFrameUpdateListener = function (when, listener, callOnce) {
    when = isSupportedUpdateListener(when) ? when : 'before';
    listener.callOnce = !!callOnce;
    this.frameUpdateHandlers[when].push(listener);

    this.refreshAutoRenderingState();
};

/**
 * Detach a listener to current instance
 * @memberof K3D.Core
 * @public
 * @param {String} when=before  Where original listener was attached to (before of after)
 * @param {Function} listener   The listener to be removed
 */
K3D.prototype.removeFrameUpdateListener = function (when, listener) {
    when = isSupportedUpdateListener(when) ? when : 'before';

    this.frameUpdateHandlers[when] = this.frameUpdateHandlers[when].filter((fn) => {
        if (fn !== listener) {
            return fn;
        }
        return false;
    });

    this.refreshAutoRenderingState();
};

module.exports = K3D;
