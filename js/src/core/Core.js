//jshint maxstatements:false

'use strict';
var viewModes = require('./lib/viewMode').viewModes,
    loader = require('./lib/Loader'),
    msgpack = require('msgpack-lite'),
    pako = require('pako'),
    screenshot = require('./lib/screenshot'),
    snapshot = require('./lib/snapshot'),
    dat = require('dat.gui'),
    resetCameraGUI = require('./lib/resetCamera'),
    detachWindowGUI = require('./lib/detachWindow'),
    fullscreen = require('./lib/fullscreen'),
    viewModeGUI = require('./lib/viewMode').viewModeGUI,
    objectGUIProvider = require('./lib/objectsGUIprovider'),
    clippingPlanesGUIProvider = require('./lib/clippingPlanesGUIProvider'),
    timeSeries = require('./lib/timeSeries');

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

    if (!provider.Helpers.validateWebGL(world.targetDOMNode)) {
        throw new Error('No WebGL');
    }

    if (typeof (provider) !== 'object') {
        throw new Error('Provider should be an object (a key-value map following convention)');
    }

    function refreshAfterObjectsChange() {
        self.getWorld().setCameraToFitScene();
        timeSeries.refreshTimeScale(self, GUI);
        Promise.all(self.rebuildSceneData()).then(self.render);
    }

    this.render = function () {
        world.render();
    };

    this.resizeHelper = function () {
        if (!self.disabling) {
            self.gui.domElement.parentNode.style['max-height'] = world.targetDOMNode.offsetHeight + 'px';
            self.Provider.Helpers.resizeListener(world);
            self.render();
        }

        dispatch(self.events.RESIZED);
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

    this.parameters = _.assign({
            viewMode: viewModes.view,
            voxelPaintColor: 0,
            menuVisibility: true,
            cameraAutoFit: true,
            gridAutoFit: true,
            gridVisible: true,
            grid: [-1, -1, -1, 1, 1, 1],
            antialias: true,
            screenshotScale: 5.0,
            clearColor: 0xffffff,
            clippingPlanes: [],
            fpsMeter: false,
            lighting: 1.5,
            time: 0.0,
            fps: 25.0,
            guiVersion: require('./../../package.json').version
        },
        parameters || {}
    );

    this.autoRendering = false;

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

        Promise.all(self.rebuildSceneData(true)).then(function () {
            self.refreshGrid();
            world.setCameraToFitScene();
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
        color = parseInt(color, 10) + 0x1000000;
        world.targetDOMNode.style.backgroundColor = '#' + color.toString(16).substr(1);
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

        world.K3DObjects.add(K3DObject);

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
        refreshAfterObjectsChange();
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
            previousValue.push(self.reload(obj, true));

            return previousValue;
        }, []);

        Promise.all(promises).then(function () {
            refreshAfterObjectsChange();
        });

        GUI.controls.__controllers.forEach(function (controller) {
            if (controller.property === 'time') {
                controller.updateDisplay();
            }
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
                objectGUIProvider(self, object.json, GUI.objects);
                world.ObjectsListJson[object.json.id] = object.json;
            });

            dispatch(self.events.OBJECT_LOADED);
            refreshAfterObjectsChange();

            return objects;
        });
    };

    /**
     * Reload object in current world
     * @memberof K3D.Core
     * @param {Object} json
     * @param {Bool} suppressRefreshAfterObjects
     */
    this.reload = function (json, suppressRefreshAfterObjects) {
        if (json.visible === false) {
            try {
                removeObjectFromScene(json.id);
                refreshAfterObjectsChange();
            } catch (e) {
                console.log(e);
            }

            return;
        }

        return loader(self, {objects: [json]}).then(function (objects) {
            objects.forEach(function (object) {
                objectGUIProvider(self, object.json, objects);
                world.ObjectsListJson[object.json.id] = object.json;
            });

            dispatch(self.events.OBJECT_LOADED);

            if (suppressRefreshAfterObjects !== true) {
                refreshAfterObjectsChange();
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
     * Get snapshot
     * @memberof K3D.Core
     * @returns {String|undefined}
     */
    this.getSnapshot = function () {
        return pako.deflate(msgpack.encode(_.values(world.ObjectsListJson)), {to: 'string', level: 9});
    };

    /**
     * Set snapshot
     * @memberof K3D.Core
     */
    this.setSnapshot = function (data) {
        var objects = msgpack.decode(pako.inflate(data));

        return self.load({objects: objects});
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
        self.setGridVisible(value);
        changeParameters.call(self, 'grid_auto_fit', value);
    });
    GUI.controls.add(self.parameters, 'gridVisible').onChange(changeParameters.bind(this, 'grid_visible'));
    viewModeGUI(GUI.controls, this);
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

    if (self.parameters.specVersion) {
        GUI.info.add({
            version: self.parameters.specVersion.substr(1)
        }, 'version').name('Spec version:');
        GUI.info.__controllers[1].__input.readOnly = true;
    }

    self.setMenuVisibility(self.parameters.menuVisibility);
    self.setTime(self.parameters.time);
    self.setGridAutoFit(self.parameters.gridAutoFit);
    self.setGridVisible(self.parameters.gridVisible);
    self.setCameraAutoFit(self.parameters.cameraAutoFit);
    self.setClippingPlanes(self.parameters.clippingPlanes);
    self.setDirectionalLightingIntensity(self.parameters.lighting);
    world.setCameraToFitScene(true);
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
    PARAMETERS_CHANGE: 'parametersChange',
    VOXELS_CALLBACK: 'voxelsCallback',
    MOUSE_MOVE: 'mouseMove',
    MOUSE_CLICK: 'mouseClick'
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
