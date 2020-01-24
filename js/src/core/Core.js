//jshint maxstatements:false

'use strict';
var viewModes = require('./lib/viewMode').viewModes,
    loader = require('./lib/Loader'),
    msgpack = require('msgpack-lite'),
    MsgpackCodec = msgpack.createCodec({preset: true}),
    pako = require('pako'),
    screenshot = require('./lib/screenshot'),
    snapshot = require('./lib/snapshot'),
    dat = require('dat.gui'),
    resetCameraGUI = require('./lib/resetCamera'),
    detachWindowGUI = require('./lib/detachWindow'),
    fullscreen = require('./lib/fullscreen'),
    viewModeGUI = require('./lib/viewMode').viewModeGUI,
    getColorLegend = require('./lib/colorMapLegend').getColorLegend,
    objectGUIProvider = require('./lib/objectsGUIprovider'),
    clippingPlanesGUIProvider = require('./lib/clippingPlanesGUIProvider'),
    timeSeries = require('./lib/timeSeries');

window.Float16Array = require('./lib/helpers/float16Array');

MsgpackCodec.addExtPacker(0x20, Float16Array, function (val) {
    return val;
});

MsgpackCodec.addExtUnpacker(0x20, function (val) {
    return Float16Array(val.buffer);
});


/**
 * @constructor Core
 * @memberof K3D
 * @param {Object} provider provider that will be used by current instance
 * @param {Node} targetDOMNode a handler for a target DOM canvas node
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
    var self = this,
        fpsMeter = null,
        objectIndex = 1,
        currentWindow = targetDOMNode.ownerDocument.defaultView || targetDOMNode.ownerDocument.parentWindow,
        world = {
            ObjectsListJson: {},
            chunkList: {},
            targetDOMNode: targetDOMNode,
            overlayDOMNode: null
        },
        listeners = {},
        listenersIndex = 0,
        dispatch = function (eventName, data) {
            if (!listeners[eventName]) {
                return false;
            }

            Object.keys(listeners[eventName]).forEach(function (key) {
                listeners[eventName][key](data);
            });

            return true;
        },
        GUI = {
            controls: null,
            objects: null
        };

    function changeParameters(key, value) {
        dispatch(self.events.PARAMETERS_CHANGE, {
            key: key,
            value: value
        });
    }

    require('style-loader?{attrs:{id: "k3d-style"}}!css-loader!./../k3d.css');

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

            timeSeries.refreshTimeScale(self, GUI);
            self.rebuildSceneData().then(self.render.bind(null, true));
        }
    };

    this.render = function (force) {
        world.render(force);
    };

    this.resizeHelper = function () {
        if (!self.disabling) {
            self.gui.domElement.parentNode.style['max-height'] = world.targetDOMNode.offsetHeight + 'px';
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
        'pointer-events: none',
        'overflow: hidden',
        'z-index: 1'
    ].join(';');

    world.targetDOMNode.appendChild(world.overlayDOMNode);

    this.GUI = GUI;
    this.parameters = _.assign({
            viewMode: viewModes.view,
            voxelPaintColor: 0,
            menuVisibility: true,
            cameraAutoFit: true,
            gridAutoFit: true,
            gridVisible: true,
            grid: [-1, -1, -1, 1, 1, 1],
            antialias: 1,
            screenshotScale: 5.0,
            renderingSteps: 1,
            clearColor: 0xffffff,
            clippingPlanes: [],
            fpsMeter: false,
            lighting: 1.5,
            time: 0.0,
            colorbarObjectId: -1,
            fps: 25.0,
            axes: ['x', 'y', 'z'],
            cameraNoRotate: false,
            cameraNoZoom: false,
            cameraNoPan: false,
            name: null,
            camera_fov: 60.0,
            autoRendering: true,
            axesHelper: 1.0,
            depthPeels: 8,
            guiVersion: require('./../../package.json').version
        },
        parameters || {}
    );

    this.autoRendering = false;

    this.startAutoPlay = function () {
        timeSeries.startAutoPlay(self);
    };

    this.stopAutoPlay = function () {
        timeSeries.stopAutoPlay(self);
    };

    this.setFps = function (fps) {
        self.parameters.fps = fps;

        GUI.controls.__controllers.forEach(function (controller) {
            if (controller.property === 'fps') {
                controller.updateDisplay();
            }
        });
    };

    this.setFpsMeter = function (state) {
        var Stats;

        if (self.parameters.fpsMeter === false) {
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
        } else {
            fpsMeter.domElement.remove();
            fpsMeter = null;
        }

        self.parameters.fpsMeter = state;
    };

    /**
     * Set autoRendering state
     * @memberof K3D.Core
     */
    this.refreshAutoRenderingState = function () {
        var handlersCount = 0;
        Object.keys(self.frameUpdateHandlers).forEach(function (when) {
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

    this.setDirectionalLightingIntensity = function (value) {
        self.parameters.lighting = Math.min(Math.max(value, 0.0), 4.0);
        self.getWorld().recalculateLights(self.parameters.lighting);
        self.render();

        GUI.controls.__controllers.forEach(function (controller) {
            if (controller.property === 'lighting') {
                controller.updateDisplay();
            }
        });
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

        GUI.controls.__controllers.forEach(function (controller) {
            if (controller.property === 'viewMode') {
                controller.updateDisplay();
            }
        });

        world.targetDOMNode.style.cursor = 'auto';
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
     * @param {String} mode
     */
    this.setMenuVisibility = function (mode) {
        self.parameters.menuVisibility = mode;
        this.gui.domElement.hidden = !mode;
    };

    this.setClippingPlanes = function (planes) {
        self.parameters.clippingPlanes = _.cloneDeep(planes);

        clippingPlanesGUIProvider(self, GUI.clippingPlanes);
        self.render();
    };

    this.setColorMapLegend = function (v) {
        var newValue = v.id || v;

        if (self.parameters.colorbarObjectId !== newValue) {
            self.parameters.colorbarObjectId = newValue;
            changeParameters('colorbar_object_id', self.parameters.colorbarObjectId);

            Object.keys(world.ObjectsListJson).forEach(function (id) {
                if (world.ObjectsListJson[id].colorLegend) {
                    world.ObjectsListJson[id].colorLegend = false;
                }
            });

            if (newValue > 0 && typeof (world.ObjectsListJson[newValue]) !== 'undefined') {
                world.ObjectsListJson[newValue].colorLegend = true;
            }

            Object.keys(GUI.objects.__folders).forEach(function (k) {
                GUI.objects.__folders[k].__controllers.forEach(function (controller) {
                    if (controller.property === 'colorLegend') {
                        controller.updateDisplay();
                    }
                });
            });
        }

        getColorLegend(self, world.ObjectsListJson[self.parameters.colorbarObjectId] || v);
    };

    /**
     * Set camera auto fit mode of K3D
     * @memberof K3D.Core
     * @param {String} mode
     */
    this.setCameraAutoFit = function (state) {
        self.parameters.cameraAutoFit = state;

        GUI.controls.__controllers.forEach(function (controller) {
            if (controller.property === 'cameraAutoFit') {
                controller.updateDisplay();
            }
        });

        if (state) {
            self.getWorld().setCameraToFitScene();
        }
    };

    /**
     * Set rendering steps of K3D
     * @memberof K3D.Core
     * @param {String} mode
     */
    this.setRenderingSteps = function (steps) {
        self.parameters.renderingSteps = steps;
    };

    /**
     * Set axes labels of plot
     * @memberof K3D.Core
     * @param {String} mode
     */
    this.setAxes = function (axesLabel) {
        self.parameters.axes = axesLabel;

        self.rebuildSceneData(true).then(function () {
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

        self.rebuildSceneData(true).then(function () {
            self.render();
        });
    };


    /**
     * Set grid auto fit mode of K3D
     * @memberof K3D.Core
     * @param {String} mode
     */
    this.setGridAutoFit = function (state) {
        self.parameters.gridAutoFit = state;
        GUI.controls.__controllers.forEach(function (controller) {
            if (controller.property === 'gridAutoFit') {
                controller.updateDisplay();
            }
        });
    };

    /**
     * Set camera lock
     * @memberof K3D.Core
     * @param {Boolean} cameraNoRotate
     * @param {Boolean} cameraNoZoom
     * @param {Boolean} cameraNoPan
     */
    this.setCameraLock = function (cameraNoRotate, cameraNoZoom, cameraNoPan) {
        self.parameters.cameraNoRotate = world.controls.noRotate = cameraNoRotate;
        self.parameters.cameraNoZoom = world.controls.noZoom = cameraNoZoom;
        self.parameters.cameraNoPan = world.controls.noPan = cameraNoPan;
    };

    /**
     * Set camera field of view
     * @memberof K3D.Core
     * @param {Float} angle
     */
    this.setCameraFOV = function (angle) {
        self.parameters.camera_fov = angle;
        world.setupCamera(null, angle);

        GUI.controls.__controllers.forEach(function (controller) {
            if (controller.property === 'camera_fov') {
                controller.updateDisplay();
            }
        });

        self.rebuildSceneData(false).then(function () {
            self.render();
        });
    };

    /**
     * Set grid auto fit mode of K3D
     * @memberof K3D.Core
     * @param {String} mode
     */
    this.setGridVisible = function (state) {
        self.parameters.gridVisible = state;
        GUI.controls.__controllers.forEach(function (controller) {
            if (controller.property === 'gridVisible') {
                controller.updateDisplay();
            }
        });

        self.refreshGrid();
        self.render();
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
     * Set grid of K3D
     * @memberof K3D.Core
     * @param {Array} vectors
     */
    this.setGrid = function (vectors) {
        self.parameters.grid = vectors;

        self.rebuildSceneData(true).then(function () {
            self.refreshGrid();
            self.render();
        });
    };

    /**
     * Set camera of K3D
     * @memberof K3D.Core
     * @param {String} mode
     */
    this.setCamera = function (array) {
        world.setupCamera(array);
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
        GUI.controls.__controllers.forEach(function (controller) {
            if (controller.property === 'voxelPaintColor') {
                controller.updateDisplay();
            }
        });
    };

    /**
     * Set clear color in renderer
     * @memberof K3D.Core
     * @param color {Number}
     */
    this.setClearColor = function (color) {
        self.parameters.clearColor = color;

        if (color >= 0) {
            color = parseInt(color, 10) + 0x1000000;
            world.targetDOMNode.style.backgroundColor = '#' + color.toString(16).substr(1);
        }
    };

    this.on = function (eventName, listener) {
        listeners[eventName] = listeners[eventName] || {};
        listeners[eventName][listenersIndex] = listener;

        listenersIndex++;
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

        }

        if (object.visible !== false) {
            world.K3DObjects.add(K3DObject);
        }

        return objectIndex++;
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
     * @param {Object} chunkList
     */
    this.setChunkList = function (json) {
        world.chunkList = json;
    };

    function removeObjectFromScene(id) {
        var object = self.Provider.Helpers.getObjectById(world, id);

        if (object) {
            world.K3DObjects.remove(object);

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
        } else {
            throw new Error('Object with id ' + id + ' dosen\'t exists');
        }
    }

    /**
     * Remove object from current world
     * @memberof K3D.Core
     * @param {String} id
     */
    this.removeObject = function (id) {
        removeObjectFromScene(id);
        delete world.ObjectsListJson[id];
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
        var timeSeriesInfo = timeSeries.getObjectsWithTimeSeriesAndMinMax(self);

        self.parameters.time = Math.min(Math.max(time, timeSeriesInfo.min), timeSeriesInfo.max);

        var promises = timeSeriesInfo.objects.reduce(function (previousValue, obj) {
            previousValue.push(self.reload(obj, null, true));

            return previousValue;
        }, []);

        GUI.controls.__controllers.forEach(function (controller) {
            if (controller.property === 'time') {
                controller.updateDisplay();
            }
        });

        return Promise.all(promises).then(function () {
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
        return loader(self, json).then(function (objects) {
            objects.forEach(function (object) {
                objectGUIProvider(self, object.json, GUI.objects, null);
                world.ObjectsListJson[object.json.id] = object.json;
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
     * @param {Bool} suppressRefreshAfterObjects
     */
    this.reload = function (json, changes, suppressRefreshAfterObjects) {
        if (json.visible === false) {
            try {
                removeObjectFromScene(json.id);
                self.refreshAfterObjectsChange(true);
            } catch (e) {
                console.log(e);
            }

            return Promise.resolve(true);
        }

        var data = {objects: [json]};

        if (changes !== null) {
            data.changes = [changes];
        }

        return loader(self, data).then(function (objects) {
            objects.forEach(function (object) {
                objectGUIProvider(self, object.json, objects, changes);
                world.ObjectsListJson[object.json.id] = object.json;
            });

            dispatch(self.events.OBJECT_LOADED);

            if (suppressRefreshAfterObjects !== true) {
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
     * @param {Object} scale
     * @param {Object} onlyCanvas
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
    this.getHTMLSnapshot = function (compression_level) {
        return snapshot.getHTMLSnapshot(this, compression_level);
    };

    /**
     * Get snapshot
     * @memberof K3D.Core
     * @returns {String|undefined}
     */
    this.getSnapshot = function (compressionLevel) {
        var chunkList = Object.keys(world.chunkList).reduce(function (p, k) {
            p[k] = world.chunkList[k].attributes;
            return p;
        }, {});

        return pako.deflate(
            msgpack.encode(
                {
                    objects: _.values(world.ObjectsListJson),
                    chunkList: chunkList
                },
                {codec: MsgpackCodec}
            ),
            {to: 'string', level: compressionLevel}
        );
    };

    /**
     * Set snapshot
     * @memberof K3D.Core
     */
    this.setSnapshot = function (data) {
        data = msgpack.decode(pako.inflate(data), {codec: MsgpackCodec});

        Object.keys(data.chunkList).forEach(function (k) {
            data.chunkList[k] = {attributes: data.chunkList[k]};
        });

        self.setChunkList(data.chunkList);

        return self.load({objects: data.objects});
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
        this.gui.destroy();
        this.autoRendering = false;

        world.K3DObjects.children.forEach(function (obj) {
            removeObjectFromScene(obj.K3DIdentifier);
            delete world.ObjectsListJson[obj.K3DIdentifier];
        });
        world.cleanup();

        if (fpsMeter) {
            fpsMeter.domElement.remove();
        }

        listeners = {};
        currentWindow.removeEventListener('resize', this.resizeHelper);
    };

    this.Provider.Initializers.Renderer.call(world, this);
    this.Provider.Initializers.Setup.call(world, this);
    this.Provider.Initializers.Camera.call(world, this);
    this.Provider.Initializers.Canvas.call(world, this);
    this.Provider.Initializers.Scene.call(world, this);

    world.controls.addEventListener('change', function () {
        if (self.frameUpdateHandlers.before.length === 0 && self.frameUpdateHandlers.after.length === 0) {
            self.render();
        }
    });

    currentWindow.addEventListener('resize', this.resizeHelper, false);

    // load toolbars
    this.gui = new dat.GUI({width: 220, autoPlace: false, scrollable: true, closeOnTop: true});

    var guiContainer = currentWindow.document.createElement('div');
    guiContainer.className = 'dg';
    guiContainer.style.cssText = [
        'position: absolute',
        'color: black',
        'top: 0',
        'right: 0',
        'z-index: 20',
        'max-height: ' + targetDOMNode.clientHeight + 'px'
    ].join(';');
    world.targetDOMNode.appendChild(guiContainer);
    guiContainer.appendChild(this.gui.domElement);

    GUI.controls = this.gui.addFolder('Controls');
    GUI.objects = this.gui.addFolder('Objects');
    GUI.info = this.gui.addFolder('Info');

    screenshot.screenshotGUI(GUI.controls, this);
    snapshot.snapshotGUI(GUI.controls, this);
    resetCameraGUI(GUI.controls, this);

    if (currentWindow === window) {
        detachWindowGUI(GUI.controls, this);

        if (fullscreen.isAvailable()) {
            fullscreen.initialize(world.targetDOMNode, GUI.controls, currentWindow);
        }
    }

    GUI.controls.add(self.parameters, 'cameraAutoFit').onChange(changeParameters.bind(this, 'camera_auto_fit'));
    GUI.controls.add(self.parameters, 'gridAutoFit').onChange(function (value) {
        self.setGridAutoFit(value);
        changeParameters.call(self, 'grid_auto_fit', value);
    });
    GUI.controls.add(self.parameters, 'gridVisible').onChange(function (value) {
        self.setGridVisible(value);
        changeParameters.call(self, 'grid_visible', value);
    });
    viewModeGUI(GUI.controls, this);
    GUI.controls.add(self.parameters, 'camera_fov').step(0.1).min(1.0).max(179).name('FOV').onChange(function (value) {
        self.setCameraFOV(value);
        changeParameters.call(self, 'camera_fov', value);
    });
    GUI.controls.add(self.parameters, 'voxelPaintColor').step(1).min(0).max(255).name('voxelColor').onChange(
        changeParameters.bind(this, 'voxel_paint_color'));
    GUI.controls.add(self.parameters, 'lighting').step(0.01).min(0).max(4).name('lighting')
        .onChange(function (value) {
            self.setDirectionalLightingIntensity(value);
            changeParameters.call(self, 'lighting', value);
        });

    timeSeries.timeSeriesGUI(GUI.controls, this, changeParameters);

    GUI.clippingPlanes = GUI.controls.addFolder('Clipping planes');

    //Info box
    GUI.info.add(self.parameters, 'guiVersion').name('Js version:');
    GUI.info.__controllers[0].__input.readOnly = true;

    if (self.parameters.backendVersion) {
        GUI.info.add({
            version: self.parameters.backendVersion
        }, 'version').name('Python version:');
        GUI.info.__controllers[1].__input.readOnly = true;
    }

    self.setMenuVisibility(self.parameters.menuVisibility);
    self.setTime(self.parameters.time);
    self.setGridAutoFit(self.parameters.gridAutoFit);
    self.setGridVisible(self.parameters.gridVisible);
    self.setCameraAutoFit(self.parameters.cameraAutoFit);
    self.setClippingPlanes(self.parameters.clippingPlanes);
    self.setDirectionalLightingIntensity(self.parameters.lighting);
    self.setColorMapLegend(self.parameters.colorbarObjectId);
    self.setAutoRendering(self.parameters.autoRendering);
    self.setCameraLock(
        self.parameters.cameraNoRotate,
        self.parameters.cameraNoZoom,
        self.parameters.cameraNoPan
    );
    self.setCameraFOV(self.parameters.camera_fov);
    self.setFps(self.parameters.fps);

    self.render();

    world.targetDOMNode.className += ' k3d-target';
}

function isSupportedUpdateListener(when) {
    return (when in {
        before: !0,
        after: !0
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
    after: []
};

K3D.prototype.events = {
    VIEW_MODE_CHANGE: 'viewModeChange',
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

    this.frameUpdateHandlers[when] = this.frameUpdateHandlers[when].filter(function (fn) {
        if (fn !== listener) {
            return fn;
        }
    });

    this.refreshAutoRenderingState();
};

module.exports = K3D;
