//jshint maxstatements:false

'use strict';
var viewModes = require('./lib/viewMode').viewModes,
    loader = require('./lib/Loader'),
    _ = require('lodash'),
    error = require('./lib/Error').error,
    resetCameraButton = require('./lib/resetCamera'),
    screenshot = require('./lib/screenshot'),
    detachWindowButton = require('./lib/detachWindow'),
    fullscreen = require('./lib/fullscreen'),
    viewModeButton = require('./lib/viewMode').viewModeButton;

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
        objectIndex = 1,
        currentWindow = targetDOMNode.ownerDocument.defaultView || targetDOMNode.ownerDocument.parentWindow,
        world = {
            ObjectsListJson: {},
            targetDOMNode: targetDOMNode,
            toolbarDOMNode: null,
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
        };

    require('style-loader?{attrs:{id: "k3d-katex"}}!css-loader!./../../node_modules/katex/dist/katex.min.css');
    require('style-loader?{attrs:{id: "k3d-style"}}!css-loader!./../k3d.css');

    if (!(this instanceof (K3D))) {
        return new K3D(provider, targetDOMNode, parameters);
    }

    if (typeof (provider) !== 'object') {
        throw new Error('Provider should be an object (a key-value map following convention)');
    }

    this.render = function () {
        world.render();
        dispatch(self.events.RENDERED);
    };

    this.resizeHelper = function () {
        if (!this.disabling) {
            self.Provider.Helpers.resizeListener(world);
            self.render();
        }

        dispatch(self.events.RESIZED);
    };

    world.toolbarDOMNode = currentWindow.document.createElement('div');
    world.toolbarDOMNode.style.cssText = [
        'position: absolute',
        'right: 0'
    ].join(';');

    world.overlayDOMNode = currentWindow.document.createElement('div');
    world.overlayDOMNode.style.cssText = [
        'position: absolute',
        'width: 100%',
        'height: 100%',
        'pointer-events: none'
    ].join(';');

    world.targetDOMNode.appendChild(world.overlayDOMNode);
    world.targetDOMNode.appendChild(world.toolbarDOMNode);

    this.parameters = _.assign({
        viewMode: viewModes.view,
        voxelPaintColor: 0,
        cameraAutoFit: true,
        gridAutoFit: true,
        grid: [-1, -1, -1, 1, 1, 1],
        antialias: true,
        clearColor: {
            color: 0xffffff,
            alpha: 1.0
        }
    }, parameters || {});

    this.autoRendering = false;

    if (typeof (this.parameters.ObjectsListJson) !== 'undefined') {
        world.ObjectsListJson = this.parameters.ObjectsListJson;
    }

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

    /**
     * Stores give provider
     * @memberof K3D.Core
     * @type {Object}
     */
    this.Provider = provider;

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
    };

    /**
     * Set camera auto fit mode of K3D
     * @memberof K3D.Core
     * @param {String} mode
     */
    this.setCameraAutoFit = function (state) {
        self.parameters.cameraAutoFit = state;
    };

    /**
     * Set grid auto fit mode of K3D
     * @memberof K3D.Core
     * @param {String} mode
     */
    this.setGridAutoFit = function (state) {
        self.parameters.gridAutoFit = state;
    };

    /**
     * Set grid of K3D
     * @memberof K3D.Core
     * @param {Array} vectors
     */
    this.setGrid = function (vectors) {
        self.parameters.grid = vectors;
        self.rebuildSceneData(true);
        world.setCameraToFitScene();
        self.render();
    };

    /**
     * Set voxelPaintColor of K3D
     * @memberof K3D.Core
     * @param {Number} color
     */
    this.setVoxelPaint = function (color) {
        self.parameters.voxelPaintColor = color;
    };

    /**
     * Set clear color in renderer
     * @memberof K3D.Core
     * @param color {Number}
     * @param alpha {Number}
     */
    this.setClearColor = function (color, alpha) {
        self.parameters.clearColor.color = color;
        self.parameters.clearColor.alpha = alpha;

        world.renderer.setClearColor(color, alpha);
        self.render();
    };

    /**
     * Update mouse position
     * @memberof K3D.Core
     * @param {Number} x
     * @param {Number} y
     */
    this.updateMousePosition = function (x, y) {
        if (self.parameters.viewMode !== viewModes.view) {
            if (self.raycast(x, y, world.camera, false, self.parameters.viewMode) && !self.autoRendering) {
                self.render();
            }
        }
    };

    /**
     * Mouse click
     * @memberof K3D.Core
     * @param {Number} x
     * @param {Number} y
     */
    this.mouseClick = function (x, y) {
        if (self.parameters.viewMode !== viewModes.view) {
            if (self.raycast(x, y, world.camera, true, self.parameters.viewMode) && !self.autoRendering) {
                self.render();
            }
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
     * Get access to camera in current world
     * @memberof K3D.Core
     * @returns {Object|undefined} - should return the "camera" if provider uses such a thing
     */
    this.getCamera = function () {
        return world.camera;
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
     * Add object to K3D objects in current world
     * @memberof K3D.Core
     * @param {Object} object
     */
    this.addObject = function (object) {
        world.K3DObjects.add(object);

        return objectIndex++;
    };

    /**
     * Remove object from current world
     * @memberof K3D.Core
     * @param {String} id
     * @param {Bool} objectsListJsonUpdating
     */
    this.removeObject = function (id, objectsListJsonUpdating) {
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

            if (object.material) {
                object.material.dispose();
                object.material = undefined;
            }

            if (object.mesh) {
                object.mesh.dispose();
                object.mesh = undefined;
            }

            if (object.texture) {
                object.texture.dispose();
                object.texture = undefined;
            }

            object = undefined;
        } else {
            throw new Error('Object with id ' + id + ' dosen\'t exists');
        }

        if (typeof (objectsListJsonUpdating) === 'undefined' || objectsListJsonUpdating === true) {
            delete world.ObjectsListJson[id];
        }

        Promise.all(self.rebuild()).then(function () {
            world.setCameraToFitScene();
            self.render();
        });
    };


    /**
     * A convenient shortcut for doing K3D.Loader(K3DInstance, json);
     * @memberof K3D.Core
     * @public
     * @param {Object} json K3D-JSON object
     * @param {Bool} objectsListJsonUpdating
     * @throws {Error} If Loader fails
     */
    this.load = function (json, objectsListJsonUpdating) {
        loader(self, json).then(function (objects) {
            if (typeof (objectsListJsonUpdating) === 'undefined' || objectsListJsonUpdating === true) {
                objects.forEach(function (object) {
                    world.ObjectsListJson[object.id] = object;
                });
            }
        });
    };

    /**
     * Reload object in current world
     * @memberof K3D.Core
     * @param {Object} json
     * @param {Bool} objectsListJsonUpdating
     */
    this.reload = function (json, objectsListJsonUpdating) {
        var object = self.Provider.Helpers.getObjectById(world, json.id);

        if (!object) {
            error('Apply Patch Object Error',
                'K3D apply patches object failed, please consult browser error console!', false);
            return;
        }

        if (typeof (objectsListJsonUpdating) === 'undefined' || objectsListJsonUpdating === true) {
            world.ObjectsListJson[json.id] = json;
        }

        self.removeObject(json.id);
        self.load({objects: [json]});
    };

    /**
     * Get Json data of  object in current world
     * @memberof K3D.Core
     * @param {Integer} id
     */
    this.getObjectJson = function (id) {
        var object = self.Provider.Helpers.getObjectById(world, id);

        if (!object) {
            error('Get Object Json Error', 'K3D get object json failed, please consult browser error console!', false);
            return;
        }

        if (object.getJson) {
            return object.getJson();
        } else {
            return null;
        }
    };

    /**
     * Rebuild scene (call after added all objects to scene)
     * @memberof K3D.Core
     */
    this.rebuild = function () {
        return self.rebuildSceneData();
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
     * @returns {Canvas|undefined}
     */
    this.getScreenshot = function () {
        return screenshot.getScreenshot(this);
    };
    /**
     * Destroy logic for current instance. Will remove listeners (browser and owned)
     * @memberof K3D.Core
     */
    this.disable = function () {
        this.disabling = true;
        this.frameUpdateHandlers.before = [];
        this.frameUpdateHandlers.after = [];
        this.autoRendering = false;
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
    screenshot.screenshotButton(world.toolbarDOMNode, this);
    resetCameraButton(world.toolbarDOMNode, this);

    if (currentWindow === window) {
        detachWindowButton(world.toolbarDOMNode, this);

        if (fullscreen.isAvailable()) {
            fullscreen.initialize(world.targetDOMNode, world.toolbarDOMNode, currentWindow);
        }
    }

    viewModeButton(world.toolbarDOMNode, this);

    world.setCameraToFitScene();
    self.render();
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
    RESIZED: 'resized'
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
