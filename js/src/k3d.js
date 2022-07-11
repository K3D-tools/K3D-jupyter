// jshint maxstatements:false

const widgets = require('@jupyter-widgets/base');
const _ = require('./lodash');
const K3D = require('./core/Core');
const TFEdit = require('./transferFunctionEditor');
const serialize = require('./core/lib/helpers/serialize');
const ThreeJsProvider = require('./providers/threejs/provider');
const CreateK3DAndLoadBinarySnapshot = require('./standalone').CreateK3DAndLoadBinarySnapshot;
const { viewModes } = require('./core/lib/viewMode');

const semverRange = require('./version').version;

const objectsList = {};
const chunkList = {};
const plotsList = [];

function runOnEveryPlot(id, cb) {
    plotsList.forEach((plot) => {
        if (plot.model.get('object_ids').indexOf(id) !== -1) {
            cb(plot, plot.K3DInstance.getObjectById(id));
        }
    });
}

const ChunkModel = widgets.WidgetModel.extend({
    defaults: _.extend(_.result({}, 'widgets.WidgetModel.prototype.defaults'), {
        _model_name: 'ChunkModel',
        _model_module: 'k3d',
        _model_module_version: semverRange,
    }),

    initialize() {
        const chunk = arguments[0];

        widgets.WidgetModel.prototype.initialize.apply(this, arguments);

        this.on('change', this._change, this);

        chunkList[chunk.id] = this;
    },

    _change() {
        const chunk = this.attributes;

        Object.keys(objectsList).forEach((id) => {
            if (objectsList[id].attributes.type === 'VoxelsGroup') {
                runOnEveryPlot(objectsList[id].attributes.id, (plot, objInstance) => {
                    objInstance.updateChunk(chunk);
                });
            }
        });
    },
}, {
    serializers: _.extend({
        voxels: serialize,
        coord: serialize,
    }, widgets.WidgetModel.serializers),
});

const ObjectModel = widgets.WidgetModel.extend({
    defaults: _.extend(_.result({}, 'widgets.WidgetModel.prototype.defaults'), {
        _model_name: 'ObjectModel',
        _view_name: 'ObjectView',
        _model_module: 'k3d',
        _view_module: 'k3d',
        _model_module_version: semverRange,
        _view_module_version: semverRange,
    }),

    initialize() {
        const obj = arguments[0];

        widgets.WidgetModel.prototype.initialize.apply(this, arguments);

        this.on('change', this._change, this);
        this.on('msg:custom', function (msg) {
            let property;

            if (msg.msg_type === 'fetch') {
                property = this.get(msg.field);

                // hack because of https://github.com/jashkenas/underscore/issues/2692
                if (_.isObject(property)) {
                    property.t = Math.random();
                }

                if (property.data && property.shape) {
                    property.compression_level = this.attributes.compression_level;
                }

                this.save(msg.field, property);
            }

            if (msg.msg_type === 'shadow_map_update' && this.get('type') === 'Volume') {
                runOnEveryPlot(this.get('id'), (plot, objInstance) => {
                    if (objInstance && objInstance.refreshLightMap) {
                        objInstance.refreshLightMap(msg.direction);
                        plot.K3DInstance.render();
                    }
                });
            }
        }, this);

        objectsList[obj.id] = this;
    },

    _change(c) {
        plotsList.forEach(function (plot) {
            plot.refreshObject(this, c.changed);
        }, this);
    },
}, {
    serializers: _.extend({
        model_matrix: serialize,
        positions: serialize,
        scalar_field: serialize,
        alpha_coef: serialize,
        shadow: serialize,
        shadow_res: serialize,
        shadow_delay: serialize,
        ray_samples_count: serialize,
        focal_plane: serialize,
        focal_length: serialize,
        gradient_step: serialize,
        color_map: serialize,
        samples: serialize,
        color_range: serialize,
        attribute: serialize,
        triangles_attribute: serialize,
        vertices: serialize,
        indices: serialize,
        colors: serialize,
        origins: serialize,
        vectors: serialize,
        opacity: serialize,
        opacities: serialize,
        point_sizes: serialize,
        point_size: serialize,
        width: serialize,
        shader: serialize,
        wireframe: serialize,
        radial_segments: serialize,
        color: serialize,
        flat_shading: serialize,
        heights: serialize,
        mesh_detail: serialize,
        voxels: serialize,
        voxels_group: serialize,
        sparse_voxels: serialize,
        space_size: serialize,
        volume: serialize,
        opacity_function: serialize,
        text: serialize,
        texture: serialize,
        binary: serialize,
        size: serialize,
        position: serialize,
        puv: serialize,
        visible: serialize,
        uvs: serialize,
        volume_bounds: serialize,
        spacings_x: serialize,
        spacings_y: serialize,
        spacings_z: serialize,
    }, widgets.WidgetModel.serializers),
});

const ObjectView = widgets.WidgetView.extend({});

const PlotModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(_.result({}, 'widgets.DOMWidgetModel.prototype.defaults'), {
        _model_name: 'PlotModel',
        _view_name: 'PlotView',
        _model_module: 'k3d',
        _view_module: 'k3d',
        _model_module_version: semverRange,
        _view_module_version: semverRange,
    }),
});

// Custom View. Renders the widget model.
const PlotView = widgets.DOMWidgetView.extend({
    render() {
        const containerEnvelope = window.document.createElement('div');
        const container = window.document.createElement('div');

        containerEnvelope.style.cssText = [
            `height:${this.model.get('height')}px`,
            'position: relative',
        ].join(';');

        container.style.cssText = [
            'width: 100%',
            'height: 100%',
            'position: relative',
        ].join(';');

        containerEnvelope.appendChild(container);
        this.el.appendChild(containerEnvelope);

        this.container = container;
        this.on('displayed', this._init, this);
    },

    remove() {
        _.pull(plotsList, this);
        this.K3DInstance.off(this.K3DInstance.events.CAMERA_CHANGE, this.cameraChangeId);
        this.K3DInstance.off(this.K3DInstance.events.OBJECT_CHANGE, this.GUIObjectChanges);
        this.K3DInstance.off(this.K3DInstance.events.PARAMETERS_CHANGE, this.GUIParametersChanges);
        this.K3DInstance.off(this.K3DInstance.events.VOXELS_CALLBACK, this.voxelsCallback);
        this.K3DInstance.off(this.K3DInstance.events.OBJECT_HOVERED, this.objectHoverCallback);
        this.K3DInstance.off(this.K3DInstance.events.OBJECT_CLICKED, this.objectClickCallback);
    },

    _init() {
        const self = this;

        this.renderPromises = [];

        plotsList.push(this);

        this.model.lastCameraSync = (new Date()).getTime();

        this.model.on('msg:custom', function (obj) {
            const { model } = this;

            if (obj.msg_type === 'fetch_screenshot') {
                this.K3DInstance.getScreenshot(this.K3DInstance.parameters.screenshotScale, obj.only_canvas)
                    .then((canvas) => {
                        const data = canvas.toDataURL().split(',')[1];

                        model.save('screenshot', data, { patch: true });
                    });
            }

            if (obj.msg_type === 'fetch_snapshot') {
                model.save('snapshot', this.K3DInstance.getHTMLSnapshot(obj.compression_level), { patch: true });
            }

            if (obj.msg_type === 'start_auto_play') {
                this.K3DInstance.startAutoPlay();
            }

            if (obj.msg_type === 'stop_auto_play') {
                this.K3DInstance.stopAutoPlay();
            }

            if (obj.msg_type === 'reset_camera') {
                this.K3DInstance.resetCamera(obj.factor);
            }

            if (obj.msg_type === 'render') {
                if (self.renderPromises.length === 0) {
                    self.K3DInstance.refreshAfterObjectsChange(false, true);
                } else {
                    Promise.all(self.renderPromises).then((values) => {
                        self.K3DInstance.refreshAfterObjectsChange(false, true);

                        if (values.length === self.renderPromises.length) {
                            self.renderPromises = [];
                        }
                    });
                }
            }
        }, this);

        this.model.on('change:camera_auto_fit', this._setCameraAutoFit, this);
        this.model.on('change:lighting', this._setDirectionalLightingIntensity, this);
        this.model.on('change:time', this._setTime, this);
        this.model.on('change:grid_auto_fit', this._setGridAutoFit, this);
        this.model.on('change:grid_visible', this._setGridVisible, this);
        this.model.on('change:grid_color', this._setGridColor, this);
        this.model.on('change:label_color', this._setLabelColor, this);
        this.model.on('change:fps_meter', this._setFpsMeter, this);
        this.model.on('change:fps', this._setFps, this);
        this.model.on('change:screenshot_scale', this._setScreenshotScale, this);
        this.model.on('change:voxel_paint_color', this._setVoxelPaintColor, this);
        this.model.on('change:background_color', this._setBackgroundColor, this);
        this.model.on('change:grid', this._setGrid, this);
        this.model.on('change:auto_rendering', this._setAutoRendering, this);
        this.model.on('change:camera', this._setCamera, this);
        this.model.on('change:camera_animation', this._setCameraAnimation, this);
        this.model.on('change:clipping_planes', this._setClippingPlanes, this);
        this.model.on('change:object_ids', this._onObjectsListChange, this);
        this.model.on('change:menu_visibility', this._setMenuVisibility, this);
        this.model.on('change:colorbar_object_id', this._setColorMapLegend, this);
        this.model.on('change:colorbar_scientific', this._setColorbarScientific, this);
        this.model.on('change:rendering_steps', this._setRenderingSteps, this);
        this.model.on('change:axes', this._setAxes, this);
        this.model.on('change:camera_no_rotate', this._setCameraLock, this);
        this.model.on('change:camera_no_zoom', this._setCameraLock, this);
        this.model.on('change:camera_no_pan', this._setCameraLock, this);
        this.model.on('change:camera_rotate_speed', this._setCameraSpeeds, this);
        this.model.on('change:camera_zoom_speed', this._setCameraSpeeds, this);
        this.model.on('change:camera_pan_speed', this._setCameraSpeeds, this);
        this.model.on('change:camera_fov', this._setCameraFOV, this);
        this.model.on('change:camera_damping_factor', this._setCameraDampingFactor, this);
        this.model.on('change:axes_helper', this._setAxesHelper, this);
        this.model.on('change:axes_helper_colors', this._setAxesHelperColors, this);
        this.model.on('change:snapshot_type', this._setSnapshotType, this);
        this.model.on('change:name', this._setName, this);
        this.model.on('change:mode', this._setViewMode, this);
        this.model.on('change:camera_mode', this._setCameraMode, this);
        this.model.on('change:manipulate_mode', this._setManipulateMode, this);

        try {
            this.K3DInstance = new K3D(ThreeJsProvider, this.container, {
                antialias: this.model.get('antialias'),
                logarithmicDepthBuffer: this.model.get('logarithmic_depth_buffer'),
                lighting: this.model.get('lighting'),
                cameraMode: this.model.get('camera_mode'),
                snapshotType: this.model.get('snapshot_type'),
                backendVersion: this.model.get('_backend_version'),
                screenshotScale: this.model.get('screenshot_scale'),
                menuVisibility: this.model.get('menu_visibility'),
                cameraAutoFit: this.model.get('camera_auto_fit'),
                cameraNoRotate: this.model.get('camera_no_rotate'),
                cameraNoZoom: this.model.get('camera_no_zoom'),
                cameraNoPan: this.model.get('camera_no_pan'),
                cameraRotateSpeed: this.model.get('camera_rotate_speed'),
                cameraZoomSpeed: this.model.get('camera_zoom_speed'),
                cameraPanSpeed: this.model.get('camera_pan_speed'),
                cameraDampingFactor: this.model.get('camera_damping_factor'),
                cameraFov: this.model.get('camera_fov'),
                colorbarObjectId: this.model.get('colorbar_object_id'),
                cameraAnimation: this.model.get('camera_animation'),
                name: this.model.get('name'),
                axes: this.model.get('axes'),
                axesHelper: this.model.get('axes_helper'),
                grid: this.model.get('grid'),
                fps: this.model.get('fps'),
                autoRendering: this.model.get('auto_rendering'),
                gridVisible: this.model.get('grid_visible'),
                gridColor: this.model.get('grid_color'),
                gridAutoFit: this.model.get('grid_auto_fit'),
                clippingPlanes: this.model.get('clipping_planes'),
                labelColor: this.model.get('label_color'),
                voxelPaintColor: this.model.get('voxel_paint_color'),
            });

            if (this.model.get('camera_auto_fit') === false) {
                this.K3DInstance.setCamera(this.model.get('camera'));
            }
        } catch (e) {
            console.log(e);
            return;
        }

        this.K3DInstance.setClearColor(this.model.get('background_color'));
        this.K3DInstance.setChunkList(chunkList);

        this.model.get('object_ids').forEach(function (id) {
            this.renderPromises.push(this.K3DInstance.load({ objects: [objectsList[id].attributes] }));
        }, this);

        this.cameraChangeId = this.K3DInstance.on(this.K3DInstance.events.CAMERA_CHANGE, (control) => {
            if (self.model._comm_live) {
                if ((new Date()).getTime() - self.model.lastCameraSync > 200) {
                    self.model.lastCameraSync = (new Date()).getTime();
                    self.model.save('camera', control, { patch: true });
                }
            }
        });

        this.GUIObjectChanges = this.K3DInstance.on(this.K3DInstance.events.OBJECT_CHANGE, (change) => {
            if (self.model._comm_live) {
                if (change.value.data && change.value.shape) {
                    change.value.compression_level = objectsList[change.id].attributes.compression_level;
                }

                if (objectsList[change.id]) {
                    objectsList[change.id].save(change.key, change.value, { patch: true });
                }
            }
        });

        this.GUIParametersChanges = this.K3DInstance.on(this.K3DInstance.events.PARAMETERS_CHANGE, (change) => {
            self.model.save(change.key, change.value, { patch: true });
        });

        this.voxelsCallback = this.K3DInstance.on(this.K3DInstance.events.VOXELS_CALLBACK, (param) => {
            if (objectsList[param.object.K3DIdentifier]) {
                objectsList[param.object.K3DIdentifier].send({ msg_type: 'click_callback', coord: param.coord });
            }
        });

        this.objectHoverCallback = this.K3DInstance.on(this.K3DInstance.events.OBJECT_HOVERED, (param) => {
            if (objectsList[param.K3DIdentifier] &&
                this.K3DInstance.parameters.viewMode === viewModes.callback) {

                objectsList[param.K3DIdentifier].send(
                    _.extend({
                        msg_type: 'hover_callback'
                    }, param)
                );
            }
        });

        this.objectClickCallback = this.K3DInstance.on(this.K3DInstance.events.OBJECT_CLICKED, (param) => {
            if (objectsList[param.K3DIdentifier] &&
                this.K3DInstance.parameters.viewMode === viewModes.callback) {

                objectsList[param.K3DIdentifier].send(
                    _.extend({
                        msg_type: 'click_callback'
                    }, param)
                );
            }
        });
    },

    _setDirectionalLightingIntensity() {
        this.K3DInstance.setDirectionalLightingIntensity(this.model.get('lighting'));
    },

    _setTime() {
        if (this.K3DInstance.parameters.time !== this.model.get('time')) {
            this.renderPromises.push(this.K3DInstance.setTime(this.model.get('time')));
        }
    },

    _setCameraAutoFit() {
        this.K3DInstance.setCameraAutoFit(this.model.get('camera_auto_fit'));
    },

    _setGridAutoFit() {
        this.K3DInstance.setGridAutoFit(this.model.get('grid_auto_fit'));
    },

    _setGridVisible() {
        this.K3DInstance.setGridVisible(this.model.get('grid_visible'));
    },

    _setGridColor() {
        this.K3DInstance.setGridColor(this.model.get('grid_color'));
    },

    _setLabelColor() {
        this.K3DInstance.setLabelColor(this.model.get('label_color'));
    },

    _setFps() {
        this.K3DInstance.setFps(this.model.get('fps'));
    },

    _setFpsMeter() {
        this.K3DInstance.setFpsMeter(this.model.get('fps_meter'));
    },

    _setScreenshotScale() {
        this.K3DInstance.setScreenshotScale(this.model.get('screenshot_scale'));
    },

    _setVoxelPaintColor() {
        this.K3DInstance.setVoxelPaint(this.model.get('voxel_paint_color'));
    },

    _setBackgroundColor() {
        this.K3DInstance.setClearColor(this.model.get('background_color'));
    },

    _setGrid() {
        this.K3DInstance.setGrid(this.model.get('grid'));
    },

    _setAutoRendering() {
        this.K3DInstance.setAutoRendering(this.model.get('auto_rendering'));
    },

    _setMenuVisibility() {
        this.K3DInstance.setMenuVisibility(this.model.get('menu_visibility'));
    },

    _setColorMapLegend() {
        this.K3DInstance.setColorMapLegend(this.model.get('colorbar_object_id'));
    },

    _setColorbarScientific() {
        this.K3DInstance.setColorbarScientific(this.model.get('colorbar_scientific'));
    },

    _setCamera() {
        this.K3DInstance.setCamera(this.model.get('camera'));
    },

    _setCameraAnimation() {
        this.K3DInstance.setCameraAnimation(this.model.get('camera_animation'));
    },

    _setRenderingSteps() {
        this.K3DInstance.setRenderingSteps(this.model.get('rendering_steps'));
    },

    _setAxes() {
        this.K3DInstance.setAxes(this.model.get('axes'));
    },

    _setName() {
        this.K3DInstance.setName(this.model.get('name'));
    },

    _setViewMode() {
        this.K3DInstance.setViewMode(this.model.get('mode'));
    },

    _setCameraMode() {
        this.K3DInstance.setCameraMode(this.model.get('camera_mode'));
    },

    _setManipulateMode() {
        this.K3DInstance.setManipulateMode(this.model.get('manipulate_mode'));
    },

    _setAxesHelper() {
        this.K3DInstance.setAxesHelper(this.model.get('axes_helper'));
    },

    _setAxesHelperColors() {
        this.K3DInstance.setAxesHelperColors(this.model.get('axes_helper_colors'));
    },

    _setSnapshotType() {
        this.K3DInstance.setSnapshotType(this.model.get('snapshot_type'));
    },

    _setCameraLock() {
        this.K3DInstance.setCameraLock(
            this.model.get('camera_no_rotate'),
            this.model.get('camera_no_zoom'),
            this.model.get('camera_no_pan'),
        );
    },

    _setCameraSpeeds() {
        this.K3DInstance.setCameraSpeeds(
            this.model.get('camera_rotate_speed'),
            this.model.get('camera_zoom_speed'),
            this.model.get('camera_pan_speed'),
        );
    },

    _setCameraFOV() {
        this.K3DInstance.setCameraFOV(this.model.get('camera_fov'));
    },

    _setCameraDampingFactor() {
        this.K3DInstance.setCameraDampingFactor(this.model.get('camera_damping_factor'));
    },

    _setClippingPlanes() {
        this.K3DInstance.setClippingPlanes(this.model.get('clipping_planes'));
    },

    _onObjectsListChange() {
        const oldObjectId = this.model.previous('object_ids');
        const newObjectId = this.model.get('object_ids');

        _.difference(oldObjectId, newObjectId).forEach(function (id) {
            this.renderPromises.push(this.K3DInstance.removeObject(id));
        }, this);

        _.difference(newObjectId, oldObjectId).forEach(function (id) {
            this.renderPromises.push(this.K3DInstance.load({ objects: [objectsList[id].attributes] }));
        }, this);
    },

    refreshObject(obj, changed) {
        if (this.model.get('object_ids').indexOf(obj.get('id')) !== -1) {
            this.renderPromises.push(this.K3DInstance.reload(objectsList[obj.get('id')].attributes, changed));
        }
    },

    processPhosphorMessage(msg) {
        widgets.DOMWidgetView.prototype.processPhosphorMessage.call(this, msg);

        switch (msg.type) {
            case 'after-attach':
                this.el.addEventListener('contextmenu', this, true);
                break;
            case 'before-detach':
                this.el.removeEventListener('contextmenu', this, true);
                break;
            case 'resize':
                this.handleResize(msg);
                break;
            default:
                break;
        }
    },

    handleEvent(event) {
        switch (event.type) {
            case 'contextmenu':
                this.handleContextMenu(event);
                break;
            default:
                widgets.DOMWidgetView.prototype.handleEvent.call(this, event);
                break;
        }
    },

    handleContextMenu(event) {
        // Cancel context menu if on renderer:
        if (this.container.contains(event.target)) {
            event.preventDefault();
            event.stopPropagation();
        }
    },

    handleResize() {
        if (this.K3DInstance) {
            this.K3DInstance.resizeHelper();
        }
    },
});

module.exports = {
    ChunkModel,
    PlotModel,
    PlotView,
    ObjectModel,
    ObjectView,
    ThreeJsProvider,
    CreateK3DAndLoadBinarySnapshot,
    TransferFunctionEditor: TFEdit.transferFunctionEditor,
    TransferFunctionModel: TFEdit.transferFunctionModel,
    TransferFunctionView: TFEdit.transferFunctionView,
    K3D,
};
