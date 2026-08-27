const fflate = require('fflate');
const LilGUI = require('lil-gui').GUI;

const msgpack = require('./lib/helpers/msgpackCodec');
const { viewModes } = require('./lib/viewMode');
const { cameraUpAxisModes } = require('./lib/cameraUpAxis');
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
const { cameraUpAxisGUI } = require('./lib/cameraUpAxis');
const manipulate = require('./lib/manipulate');
const { getColorLegend } = require('./lib/colorMapLegend');
const objectsGUIProvider = require('./lib/objectsGUIprovider');
const clippingPlanesGUIProvider = require('./lib/clippingPlanesGUIProvider');
const timeSeries = require('./lib/timeSeries');
const { base64ToArrayBuffer } = require('./lib/helpers/buffer');
const { error: errorOverlay } = require('./lib/Error');

const Float16Array = require('./lib/helpers/float16Array');

window.Float16Array = Float16Array;

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
    let removeFullscreenListener = null;
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
            width: 220, autoPlace: false, title: 'K3D panel',
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
                // Keep the remover: initializeGUI runs again every time the menu is re-shown,
                // and the listener it installs sits on the main window and captures this
                // instance.
                removeFullscreenListener = fullscreen.initialize(
                    world.targetDOMNode,
                    GUI.controls,
                    currentWindow,
                    self,
                );
            }
        }

        GUI.controls.add(self.parameters, 'cameraAutoFit').onChange(changeParameters.bind(
            self,
            'camera_auto_fit',
        ));
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
        GUI.controls.add(self.parameters, 'depthPeels').step(1).min(0).max(16)
            .onChange((value) => {
                self.setDepthPeels(value);
                changeParameters.call(self, 'depth_peels', value);
            });
        GUI.controls.add(self.parameters, 'renderer', ['simple', 'advanced', 'cinematic']).listen()
            .onChange((value) => {
                self.setRenderer(value);
                // setRenderer can refuse the switch, so sync the surviving mode, not the requested one
                changeParameters.call(self, 'renderer', self.parameters.renderer);
            });
        // Python resolves catalog names to arrays, so the resolved value comes back as
        // an object - the wire dict carries the name for display. Arrays without one
        // show as 'custom'. Without a kernel nothing can resolve a catalog name, so a
        // standalone offers only what it can regenerate: the procedural presets, any
        // sideloaded maps (window.k3dEnvironments) and the map baked into the snapshot -
        // that one is kept aside, or switching away from it would lose the pixels.
        const bakedEnv = (self.parameters.environment && self.parameters.environment.data
            && self.parameters.environment.name) ? self.parameters.environment : null;
        const environmentOptions = (function () {
            const options = ['neutral', 'studio', 'outdoor'];

            if (self.parameters.standaloneGUI) {
                if (typeof (window) !== 'undefined' && window.k3dEnvironments) {
                    Object.keys(window.k3dEnvironments).forEach((name) => {
                        if (options.indexOf(name) === -1) {
                            options.push(name);
                        }
                    });
                }

                if (bakedEnv && options.indexOf(bakedEnv.name) === -1) {
                    options.push(bakedEnv.name);
                }
            } else {
                options.push(
                    'autoshop_01',
                    'brown_photostudio_02',
                    'burnt_warehouse',
                    'moonless_golf',
                    'venice_sunset',
                );
            }

            options.push('custom');

            return options;
        }());
        const environmentGUIName = function () {
            const env = self.parameters.environment;

            if (typeof (env) === 'string') {
                return env;
            }
            if (env && env.name && environmentOptions.indexOf(env.name) !== -1) {
                return env.name;
            }

            return 'custom';
        };
        const environmentProxy = { environment: environmentGUIName() };

        self.refreshEnvironmentGUI = function () {
            environmentProxy.environment = environmentGUIName();
        };

        const environmentControls = [];

        environmentControls.push(GUI.controls.add(environmentProxy, 'environment', environmentOptions)
            .listen()
            .onChange((value) => {
                if (value === 'custom') {
                    // a label for an array map, not a choice
                    self.refreshEnvironmentGUI();
                    return;
                }
                if (bakedEnv && value === bakedEnv.name) {
                    // back to the map baked into this snapshot
                    self.setEnvironment(bakedEnv);
                    changeParameters.call(self, 'environment', value);
                    return;
                }
                self.setEnvironment(value);
                changeParameters.call(self, 'environment', value);
            }));
        environmentControls.push(GUI.controls.add(self.parameters, 'environmentRotation')
            .step(0.01).min(0).max(2 * Math.PI)
            .listen()
            .onChange((value) => {
                self.setEnvironmentRotation(value);
                changeParameters.call(self, 'environment_rotation', value);
            }));
        const aoControls = [];

        aoControls.push(GUI.controls.add(self.parameters, 'aoRadius')
            .step(0.005).min(0.005).max(0.5)
            .listen()
            .onChange((value) => {
                self.setAORadius(value);
                changeParameters.call(self, 'ao_radius', value);
            }));
        aoControls.push(GUI.controls.add(self.parameters, 'aoStrength')
            .step(0.05).min(0).max(5)
            .listen()
            .onChange((value) => {
                self.setAOStrength(value);
                changeParameters.call(self, 'ao_strength', value);
            }));

        const cinematicControls = [];

        // slider max is an interactive comfort bound; the trait accepts more
        cinematicControls.push(GUI.controls.add(self.parameters, 'cinematicSamples')
            .step(1).min(1).max(1024)
            .listen()
            .onChange((value) => {
                self.setCinematicSamples(value);
                changeParameters.call(self, 'cinematic_samples', value);
            }));
        cinematicControls.push(GUI.controls.add(self.parameters, 'cinematicBounces')
            .step(1).min(1).max(16)
            .listen()
            .onChange((value) => {
                self.setCinematicBounces(value);
                changeParameters.call(self, 'cinematic_bounces', value);
            }));
        cinematicControls.push(GUI.controls.add(self.parameters, 'cinematicGlossyFilter')
            .step(0.05).min(0.0).max(1.0)
            .listen()
            .onChange((value) => {
                self.setCinematicGlossyFilter(value);
                changeParameters.call(self, 'cinematic_glossy_filter', value);
            }));

        self.refreshRendererGUI = function () {
            const mode = self.parameters.renderer;

            environmentControls.forEach((control) => {
                control.show(mode === 'advanced' || mode === 'cinematic');
            });
            aoControls.forEach((control) => {
                control.show(mode === 'advanced');
            });
            cinematicControls.forEach((control) => {
                control.show(mode === 'cinematic');
            });
        };
        self.refreshRendererGUI();
        GUI.controls.add(self.parameters, 'toneMapping', ['none', 'agx', 'aces']).listen()
            .onChange((value) => {
                self.setToneMapping(value);
                changeParameters.call(self, 'tone_mapping', value);
            });
        viewModeGUI(GUI.controls, self);
        cameraModeGUI(GUI.controls, self);
        cameraUpAxisGUI(GUI.controls, self);
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

            // Deleting an object in manipulate mode would otherwise leave its gizmo in the scene.
            if (object.transformControls) {
                object.transformControls.detach();
                world.scene.remove(object.transformControls);
                object.transformControls.dispose();
                delete object.transformControls;
            }

            // Voxels, vectors and labels nest their meshes in a group, which carries no geometry
            // or material of its own, so disposing only the top level released nothing for them.
            object.traverse((node) => {
                if (node.geometry) {
                    node.geometry.dispose();
                }

                if (node.material) {
                    [].concat(node.material).forEach((material) => {
                        if (material.map) {
                            material.map.dispose();
                        }

                        material.dispose();
                    });
                }
            });

            if (object.mesh) {
                object.mesh.dispose();
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
        if (self.parameters.renderOnChange || force) {
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
        'user-select: none',
        '-webkit-user-select: none',
    ].join(';');

    this.GUI = GUI;
    this.parameters = _.assignWith(
        {
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
            sliceViewerDirection: 'z',
            sliceViewerObjectId: -1,
            sliceViewerMaskObjectIds: [],
            colorbarObjectId: -1,
            colorbarScientific: false,
            fps: 25.0,
            time: 0.0,
            timeSpeed: 1.0,
            timeInterpolation: true,
            axes: ['x', 'y', 'z'],
            minimumFps: -1,
            cameraNoRotate: false,
            cameraNoZoom: false,
            cameraNoPan: false,
            cameraRotateSpeed: 1.0,
            cameraZoomSpeed: 1.2,
            cameraPanSpeed: 0.3,
            cameraDampingFactor: 0.0,
            name: null,
            cameraFov: 60.0,
            cameraUpAxis: cameraUpAxisModes.none,
            cameraAnimation: {},
            renderOnChange: true,
            axesHelper: 1.0,
            axesHelperColors: [0xff0000, 0x00ff00, 0x0000ff],
            depthPeels: 0,
            renderer: 'simple',
            environment: 'neutral',
            environmentRotation: 0.0,
            toneMapping: 'none',
            aoRadius: 0.07,
            aoStrength: 1.8,
            cinematicSamples: 64,
            cinematicBounces: 6,
            cinematicGlossyFilter: 0.25,
            snapshotType: 'full',
            customData: null,
            additionalJsCode: '',
            hiddenObjectIds: [],
            guiVersion: require('../../package.json').version,
        },
        parameters || {},
        (objValue, srcValue) => (typeof (srcValue) === 'undefined' ? objValue : srcValue),
    );

    let prevDepthPeels = self.parameters.depthPeels;

    this.setMinimumFps = function (fpsTarget) {
        self.parameters.minimumFps = fpsTarget;
    };

    this.startAutoPlay = function () {
        timeSeries.startAutoPlay(self, changeParameters);
    };

    this.stopAutoPlay = function () {
        timeSeries.stopAutoPlay(self);
    };

    this.setFps = function (fps) {
        self.parameters.fps = fps;

        if (GUI.controls) {
            GUI.controls.controllersMap.fps.updateDisplay();
        }
    };

    this.setTimeSpeed = function (timeSpeed) {
        self.parameters.timeSpeed = timeSpeed;

        if (GUI.controls) {
            GUI.controls.controllersMap.timeSpeed.updateDisplay();
        }
    };

    this.getTimeSeriesInfo = function () {
        const info = timeSeries.getObjectsWithTimeSeriesAndMinMax(self);

        return {
            min: info.min,
            max: info.max,
            times: timeSeries.getTimeSeriesTimes(self),
        };
    };

    this.stepFrame = function (step) {
        const times = timeSeries.getTimeSeriesTimes(self);

        if (times.length === 0) {
            return self.parameters.time;
        }

        let nearest = 0;

        for (let i = 1; i < times.length; i++) {
            if (Math.abs(times[i] - self.parameters.time)
                < Math.abs(times[nearest] - self.parameters.time)) {
                nearest = i;
            }
        }

        const index = Math.min(Math.max(nearest + step, 0), times.length - 1);

        self.setTime(times[index]);

        return self.parameters.time;
    };

    this.setTimeInterpolation = function (timeInterpolation) {
        self.parameters.timeInterpolation = timeInterpolation;

        // Re-apply the current time so the change shows without waiting for the next tick.
        self.setTime(self.parameters.time);
    };

    this.setAdditionalJsCode = function (additionalJsCode) {
        self.parameters.additionalJsCode = additionalJsCode;
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
    };

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
     * Whether adding or updating an object draws a frame on its own. It has never had
     * anything to do with a render loop - K3D draws only when something changed.
     * @memberof K3D.Core
     * @param {Bool} flag
     */
    this.setRenderOnChange = function (flag) {
        self.parameters.renderOnChange = flag;
    };

    /**
     * @deprecated renamed to setRenderOnChange in 3.0.0
     * @memberof K3D.Core
     * @param {Bool} flag
     */
    this.setAutoRendering = function (flag) {
        self.setRenderOnChange(flag);
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
            if (removeFullscreenListener) {
                removeFullscreenListener();
                removeFullscreenListener = null;
            }

            Object.keys(self.gui_map || {}).forEach((id) => {
                const folder = self.gui_map[id];

                if (folder && folder.listenersId) {
                    self.off(self.events.OBJECT_REMOVED, folder.listenersId);
                }
            });

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

    this.setSliceViewerMaskObjects = function (objectIds) {
        objectIds = _.cloneDeep(objectIds);
        self.parameters.sliceViewerMaskObjectIds.length = 0;

        objectIds.forEach((o) => {
            self.parameters.sliceViewerMaskObjectIds.push(o);
        });

        Object.keys(world.ObjectsListJson).forEach((id) => {
            const flag = self.parameters.sliceViewerMaskObjectIds.indexOf(parseInt(id, 10)) !== -1;

            world.ObjectsListJson[id].volumeSliceMask = flag;

            objectsGUIProvider.update(self, world.ObjectsListJson[id], GUI.objects, {
                volumeSliceMask: flag,
            });
        });

        changeParameters('slice_viewer_mask_object_ids', self.parameters.sliceViewerMaskObjectIds);

        if (world.controls.reslice) {
            world.controls.reslice();
        }

        world.controls.update();
        self.render();
    };

    this.setColorbarScientific = function (flag) {
        self.parameters.colorbarScientific = flag;
        self.render();
    };

    this.setSliceViewerDirection = function (direction) {
        self.parameters.sliceViewerDirection = direction;
        world.controls.update();
        self.render();
    };

    this.setSliceViewer = function (v) {
        const newValue = v.id || v;

        if (self.parameters.sliceViewerObjectId !== newValue) {
            self.parameters.sliceViewerObjectId = newValue;
            changeParameters('slice_viewer_object_id', self.parameters.sliceViewerObjectId);

            Object.keys(world.ObjectsListJson).forEach((id) => {
                if (world.ObjectsListJson[id].sliceViewer) {
                    world.ObjectsListJson[id].sliceViewer = false;
                }
            });

            if (newValue > 0 && typeof (world.ObjectsListJson[newValue]) !== 'undefined') {
                world.ObjectsListJson[newValue].sliceViewer = true;
            }

            if (GUI.objects) {
                Object.keys(GUI.objects.folders).forEach((k) => {
                    GUI.objects.folders[k].controllers.forEach((controller) => {
                        if (controller.property === 'sliceViewer') {
                            controller.updateDisplay();
                        }
                    });
                });
            }

            world.controls.update();
            self.render();
        }
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
    };

    /**
     * Set camera up axis
     * @memberof K3D.Core
     * @param {String} axis
     */
    this.setCameraUpAxis = function (axis) {
        self.parameters.cameraUpAxis = axis;

        self.getWorld().changeControls(true);

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'cameraUpAxis') {
                    controller.updateDisplay();
                }
            });
        }

        self.rebuildSceneData(false).then(() => {
            self.render();
        });
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
     * Set depth peels count of K3D
     * @memberof K3D.Core
     * @param {Number} count
     */
    this.setDepthPeels = function (count) {
        const objectsPromieses = [];

        self.parameters.depthPeels = count;

        if (GUI.controls) {
            GUI.controls.controllers.forEach((controller) => {
                if (controller.property === 'depthPeels') {
                    controller.updateDisplay();
                }
            });
        }

        if ((prevDepthPeels === 0 && count > 0)
            || (prevDepthPeels > 0 && count === 0)) {
            _.values(world.ObjectsListJson).forEach((json) => {
                // blending and the peel shader hook are chosen when the material is built, so
                // every object has to be built again. Dropping it from the scene first is what
                // makes that happen: the loader updates in place whenever it still finds the
                // object and has any change to apply, and a time series always has one.
                try {
                    removeObjectFromScene(json.id);
                } catch (e) {
                    // nothing
                }

                objectsPromieses.push(self.reload(json, null));
            });
        }

        prevDepthPeels = count;

        return Promise.all(objectsPromieses).then(() => {
            self.render();
        });
    };

    /**
     * Set renderer mode of K3D
     * @memberof K3D.Core
     * @param {String} mode 'simple' or 'advanced'
     */
    this.setRenderer = function (mode) {
        if (mode !== 'simple' && mode !== 'advanced' && mode !== 'cinematic') {
            // this travels in snapshots between versions, so an unknown value degrades
            console.warn(`K3D: unknown renderer "${mode}", falling back to "simple"`);
            mode = 'simple';
        }

        // an unsupported path tracer is refused, never downgraded to another mode; a snapshot
        // already in cinematic has no previous mode to keep, so render() reports the failure
        if (mode === 'cinematic' && self.parameters.renderer !== 'cinematic') {
            const reason = world.cinematicUnsupportedReason
                ? world.cinematicUnsupportedReason()
                : 'the current provider has no cinematic backend';

            if (reason !== null) {
                errorOverlay('Cinematic Error', `The cinematic renderer cannot start: ${reason}.`, false);
                changeParameters('renderer', self.parameters.renderer);

                if (self.refreshRendererGUI) {
                    self.refreshRendererGUI();
                }

                return;
            }
        }

        self.parameters.renderer = mode;

        if (self.refreshRendererGUI) {
            self.refreshRendererGUI();
        }

        world.applyRendererMode(self);
        self.render();
    };

    /**
     * Set environment of K3D
     * @memberof K3D.Core
     * @param {String|Object} environment preset name or an equirect array
     */
    this.setEnvironment = function (environment) {
        self.parameters.environment = environment;

        if (self.refreshEnvironmentGUI) {
            self.refreshEnvironmentGUI();
        }

        world.applyRendererMode(self);
        self.render();
    };

    /**
     * Set environment rotation of K3D
     * @memberof K3D.Core
     * @param {Number} rotation around the up axis, radians
     */
    this.setEnvironmentRotation = function (rotation) {
        self.parameters.environmentRotation = rotation;
        world.applyRendererMode(self);
        self.render();
    };

    /**
     * Set ambient occlusion radius (fraction of the scene diagonal)
     * @memberof K3D.Core
     * @param {Number} radius
     */
    this.setAORadius = function (radius) {
        self.parameters.aoRadius = radius;
        self.render();
    };

    /**
     * Set ambient occlusion strength (shadow-deepening exponent)
     * @memberof K3D.Core
     * @param {Number} strength
     */
    this.setAOStrength = function (strength) {
        self.parameters.aoStrength = strength;
        self.render();
    };

    /**
     * Set the sample budget of the cinematic renderer (screenshot/headless)
     * @memberof K3D.Core
     * @param {Number} samples
     */
    this.setCinematicSamples = function (samples) {
        self.parameters.cinematicSamples = samples;
        self.render();
    };

    /**
     * Set the light bounce count of the cinematic renderer
     * @memberof K3D.Core
     * @param {Number} bounces
     */
    this.setCinematicBounces = function (bounces) {
        self.parameters.cinematicBounces = bounces;
        self.render();
    };

    /**
     * Set how strongly the cinematic renderer widens glossy lobes after a rough bounce
     * @memberof K3D.Core
     * @param {Number} factor 0 leaves them sharp
     */
    this.setCinematicGlossyFilter = function (factor) {
        self.parameters.cinematicGlossyFilter = factor;
        self.render();
    };

    /**
     * Set tone mapping of K3D
     * @memberof K3D.Core
     * @param {String} name 'none', 'agx' or 'aces'
     */
    this.setToneMapping = function (name) {
        if (name !== 'none' && name !== 'agx' && name !== 'aces') {
            console.warn(`K3D: unknown tone_mapping "${name}", falling back to "none"`);
            name = 'none';
        }

        self.parameters.toneMapping = name;
        world.applyToneMapping(name);
        self.render();
    };

    /**
     * Set renderable objects ids
     * @memberof K3D.Core
     */
    this.setHiddenObjectIds = function (list) {
        self.parameters.hiddenObjectIds = list;

        self.render();
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
     * Current event subscriptions, so a replacement instance can take them over.
     * detachWindow builds a new Core and copies it onto the old object; its on/off/dispatch
     * close over a fresh listeners map, so the subscriptions have to be carried across.
     * @memberof K3D.Core
     * @returns {Object}
     */
    this.getListeners = function () {
        return { listeners, listenersIndex };
    };

    /**
     * Take over subscriptions captured from another instance via getListeners().
     * @memberof K3D.Core
     * @param {Object} state
     */
    this.adoptListeners = function (state) {
        if (!state || !state.listeners) {
            return;
        }

        Object.keys(state.listeners).forEach((eventName) => {
            listeners[eventName] = Object.assign(listeners[eventName] || {}, state.listeners[eventName]);
        });

        // Keep handing out fresh ids so an adopted id is never reused.
        listenersIndex = Math.max(listenersIndex, state.listenersIndex);
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

        // skip non-webgl objects
        if (object.type !== 'Text' && object.type !== 'Text2d') {
            world.K3DObjects.add(K3DObject);
        }

        // registered here, at add time - not only in reload's .then. Two overlapping
        // creates for the same id (GUI change + model echo) must see each other, or the
        // earlier instance stays in the scene as an unremovable orphan
        if (typeof (object.id) !== 'undefined') {
            world.ObjectsById[object.id] = K3DObject;
        }

        objectIndex += 1;

        self.heavyOperationSync = true;

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

            world.setupCamera(newCamera.json.camera, null, true);
        }

        if (GUI.controls) {
            GUI.controls.controllersMap.time.updateDisplay();
        }

        dispatch(self.events.TIME_CHANGE, self.parameters.time);

        return Promise.all(promises).then(() => self.refreshAfterObjectsChange(true));
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
                if (!object) { return; }

                objectsGUIProvider.update(self, object.json, GUI.objects, null);

                world.ObjectsListJson[object.json.id] = object.json;

                // a concurrent create may have evicted this instance already - never
                // point the registry back at a disposed object
                if (!world.ObjectsById[object.json.id] || world.ObjectsById[object.json.id] === object.obj) {
                    world.ObjectsById[object.json.id] = object.obj;
                }

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
            try {
                removeObjectFromScene(json.id);
            } catch (e) {
                // nothing
            }

            // the render has to follow the removal: it is the last frame drawn, so rendering
            // first leaves the object on screen until something else triggers a new one
            if (timeSeriesReload !== true) {
                objectsGUIProvider.update(self, json, GUI.objects, changes);
                self.refreshAfterObjectsChange(true);
            }

            return Promise.resolve(true);
        }

        const data = { objects: [json] };

        if (changes !== null) {
            data.changes = [changes];
        }

        return loader(self, data).then((objects) => {
            objects.forEach((object) => {
                if (!object) { return; } // Loader could not create it; already reported

                if (timeSeriesReload !== true) {
                    objectsGUIProvider.update(self, object.json, GUI.objects, changes);
                }

                world.ObjectsListJson[object.json.id] = object.json;

                // same eviction rule as in load: a newer create wins the registry
                if (!world.ObjectsById[object.json.id] || world.ObjectsById[object.json.id] === object.obj) {
                    world.ObjectsById[object.json.id] = object.obj;
                }

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
            ),
            { level: compressionLevel },
        );
    };

    /**
     * Set snapshot
     * @memberof K3D.Core
     */
    this.setSnapshot = function (data) {
        try {
            if (typeof (data) === 'string') {
                data = fflate.unzlibSync(new Uint8Array(base64ToArrayBuffer(data)));
            }

            if (data instanceof Uint8Array) {
                data = msgpack.decode(data);
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

            return self.load({ objects: data.objects }).then(() => self.refreshAfterObjectsChange(
                false,
                true,
            ));
        } catch (error) {
            console.error('K3D: Failed to set snapshot:', error.message);
            throw new Error(`Invalid snapshot data: ${error.message}`);
        }
    };

    /**
     * Extract snapshot
     * @memberof K3D.Core
     * @param {String} data
     * @returns {Object|undefined}
     */
    this.extractSnapshot = function (data) {
        return data.match(/var data(?:_[^\s=']{1,64})? = '([^']*)';/mi);
    };

    /**
     * Destroy logic for current instance. Will remove listeners (browser and owned)
     * @memberof K3D.Core
     */
    this.disable = function () {
        // The autoplay loop reschedules itself through requestAnimationFrame, so it outlives the
        // instance and keeps driving setTime on a destroyed GUI and a lost GL context. Stop it
        // before the GUI goes away, since stopAutoPlay relabels its button.
        self.stopAutoPlay();

        this.disabling = true;
        if (this.gui) {
            this.gui.destroy();
        }

        Object.keys(world.ObjectsListJson).forEach((K3DIdentifier) => {
            removeObjectFromScene(K3DIdentifier);
            delete world.ObjectsListJson[K3DIdentifier];
        });

        // Reachable from the Canvas initializer, before Scene defines cleanup.
        if (world.cleanup) {
            world.cleanup();
        }

        if (fpsMeter) {
            fpsMeter.domElement.remove();
            fpsMeter = null;
        }

        if (removeFullscreenListener) {
            removeFullscreenListener();
            removeFullscreenListener = null;
        }

        listeners = {};

        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }

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

    this.resizeObserver = new ResizeObserver(() => {
        this.resizeHelper();
    });
    this.resizeObserver.observe(targetDOMNode);

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
    self.setFps(self.parameters.fps);
    self.setTimeSpeed(self.parameters.timeSpeed);
    self.setAdditionalJsCode(self.parameters.additionalJsCode);
    self.setGridAutoFit(self.parameters.gridAutoFit);
    self.setGridVisible(self.parameters.gridVisible);
    self.setGrid(self.parameters.grid);
    self.setDepthPeels(self.parameters.depthPeels);
    self.setRenderer(self.parameters.renderer);
    self.setToneMapping(self.parameters.toneMapping);
    self.setCameraAutoFit(self.parameters.cameraAutoFit);
    self.setCameraDampingFactor(self.parameters.cameraDampingFactor);
    self.setCameraUpAxis(self.parameters.cameraUpAxis);
    self.setClippingPlanes(self.parameters.clippingPlanes);
    self.setSliceViewer(self.parameters.sliceViewerObjectId);
    self.setSliceViewerMaskObjects(self.parameters.sliceViewerMaskObjectIds);
    self.setSliceViewerDirection(self.parameters.sliceViewerDirection);
    self.setDirectionalLightingIntensity(self.parameters.lighting);
    self.setColorMapLegend(self.parameters.colorbarObjectId);
    self.setColorbarScientific(self.parameters.colorbarScientific);
    self.setRenderOnChange(self.parameters.renderOnChange);
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
    self.setViewMode(self.parameters.viewMode);
    self.setHiddenObjectIds(self.parameters.hiddenObjectIds);
    self.setFpsMeter(self.parameters.fpsMeter);

    self.MsgpackCodec = msgpack.codec;
    self.msgpack = msgpack;
    self.serialize = serialize;

    self.render();

    world.targetDOMNode.className += ' k3d-target';
}

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
    // Not PARAMETERS_CHANGE: that one is written back to the model, so a time set from the
    // kernel would be echoed back. Fires per frame during playback, so listeners stay cheap.
    TIME_CHANGE: 'timeChange',
    AUTO_PLAY_CHANGE: 'autoPlayChange',
    VOXELS_CALLBACK: 'voxelsCallback',
    MOUSE_MOVE: 'mouseMove',
    MOUSE_CLICK: 'mouseClick',
};

module.exports = K3D;
