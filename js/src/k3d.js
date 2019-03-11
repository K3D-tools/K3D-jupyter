'use strict';

var widgets = require('@jupyter-widgets/base'),
    K3D = require('./core/Core'),
    serialize = require('./core/lib/helpers/serialize'),
    ThreeJsProvider = require('./providers/threejs/provider'),
    PlotModel,
    PlotView,
    ObjectModel,
    ObjectView,
    semverRange = require('./version').EXTENSION_SPEC_VERSION,
    objectsList = {},
    plotsList = [];

require('es6-promise');

ObjectModel = widgets.WidgetModel.extend({
    defaults: _.extend(_.result({}, 'widgets.WidgetModel.prototype.defaults'), {
        _model_name: 'ObjectModel',
        _view_name: 'ObjectView',
        _model_module: 'k3d',
        _view_module: 'k3d',
        _model_module_version: semverRange,
        _view_module_version: semverRange
    }),

    runOnEveryPlot: function (cb) {
        plotsList.forEach(function (plot) {
            if (plot.model.get('object_ids').indexOf(this.get('id')) !== -1) {
                cb(plot, plot.K3DInstance.getObjectById(this.get('id')));
            }
        }, this);
    },

    initialize: function () {
        var obj = arguments[0];

        widgets.WidgetModel.prototype.initialize.apply(this, arguments);

        this.on('change', this._change, this);
        this.on('msg:custom', function (msg) {
            var obj;

            if (msg.msg_type === 'fetch') {
                obj = this.get(msg.field);

                // hack because of https://github.com/jashkenas/underscore/issues/2692
                if (_.isObject(obj)) {
                    obj.t = Math.random();
                }

                this.save(msg.field, obj);
            }

            if (msg.msg_type === 'shadow_map_update' && this.get('type') === 'Volume') {
                this.runOnEveryPlot(function (plot, objInstance) {
                    objInstance.refreshLightMap(msg.direction);
                    plot.K3DInstance.render();
                });
            }
        }, this);

        objectsList[obj.id] = this;
    },

    _change: function () {
        plotsList.forEach(function (plot) {
            plot.refreshObject(this);
        }, this);
    }
}, {
    serializers: _.extend({
        model_matrix: serialize,
        positions: serialize,
        scalar_field: serialize,
        color_map: serialize,
        attribute: serialize,
        vertices: serialize,
        indices: serialize,
        colors: serialize,
        origins: serialize,
        vectors: serialize,
        heights: serialize,
        voxels: serialize,
        voxels_group: serialize,
        sparse_voxels: serialize,
        space_size: serialize,
        volume: serialize
    }, widgets.WidgetModel.serializers)
});

ObjectView = widgets.WidgetView.extend({});

PlotModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(_.result({}, 'widgets.DOMWidgetModel.prototype.defaults'), {
        _model_name: 'PlotModel',
        _view_name: 'PlotView',
        _model_module: 'k3d',
        _view_module: 'k3d',
        _model_module_version: semverRange,
        _view_module_version: semverRange
    })
});

// Custom View. Renders the widget model.
PlotView = widgets.DOMWidgetView.extend({
    render: function () {
        var containerEnvelope = window.document.createElement('div'),
            container = window.document.createElement('div');

        containerEnvelope.style.cssText = [
            'height:' + this.model.get('height') + 'px',
            'position: relative'
        ].join(';');

        container.style.cssText = [
            'width: 100%',
            'height: 100%',
            'position: relative'
        ].join(';');

        containerEnvelope.appendChild(container);
        this.el.appendChild(containerEnvelope);

        this.container = container;
        this.on('displayed', this._init, this);
    },

    remove: function () {
        _.pull(plotsList, this);
        this.K3DInstance.off(this.K3DInstance.events.CAMERA_CHANGE, this.cameraChangeId);
        this.K3DInstance.off(this.K3DInstance.events.OBJECT_CHANGE, this.GUIObjectChanges);
        this.K3DInstance.off(this.K3DInstance.events.PARAMETERS_CHANGE, this.GUIParametersChanges);
        this.K3DInstance.off(this.K3DInstance.events.VOXELS_CALLBACK, this.voxelsCallback);
    },

    _init: function () {
        var self = this;

        plotsList.push(this);

        this.model.on('msg:custom', function (obj) {
            var model = this.model;

            if (obj.msg_type === 'fetch_screenshot') {
                this.K3DInstance.getScreenshot(this.K3DInstance.parameters.screenshotScale, obj.only_canvas)
                    .then(function (canvas) {
                        var data = canvas.toDataURL().split(',')[1];

                        model.save('screenshot', data);
                    });
            }
        }, this);
        this.model.on('change:camera_auto_fit', this._setCameraAutoFit, this);
        this.model.on('change:lighting', this._setDirectionalLightingIntensity, this);
        this.model.on('change:time', this._setTime, this);
        this.model.on('change:grid_auto_fit', this._setGridAutoFit, this);
        this.model.on('change:grid_visible', this._setGridVisible, this);
        this.model.on('change:fps_meter', this._setFpsMeter, this);
        this.model.on('change:screenshot_scale', this._setScreenshotScale, this);
        this.model.on('change:voxel_paint_color', this._setVoxelPaintColor, this);
        this.model.on('change:background_color', this._setBackgroundColor, this);
        this.model.on('change:grid', this._setGrid, this);
        this.model.on('change:camera', this._setCamera, this);
        this.model.on('change:clipping_planes', this._setClippingPlanes, this);
        this.model.on('change:object_ids', this._onObjectsListChange, this);
        this.model.on('change:menu_visibility', this._setMenuVisibility, this);
        this.model.on('change:colorbar_object_id', this._setColorMapLegend, this);

        try {
            this.K3DInstance = new K3D(ThreeJsProvider, this.container, {
                antialias: this.model.get('antialias'),
                specVersion: this.model.get('_view_module_version'),
                backendVersion: this.model.get('_backend_version'),
                screenshotScale: this.model.get('screenshot_scale'),
                menuVisibility: this.model.get('menu_visibility'),
                grid: this.model.get('grid')
            });
        } catch (e) {
            console.log(e);
            return;
        }

        this.objectsChangesQueue = [];
        this.objectsChangesQueueRun = false;

        this.K3DInstance.setClearColor(this.model.get('background_color'));

        this._setCameraAutoFit();
        this._setGridAutoFit();
        this._setMenuVisibility();
        this._setVoxelPaintColor();

        this.model.get('object_ids').forEach(function (id) {
            this.objectsChangesQueue.push({id: id, operation: 'insert'});
        }, this);

        if (this.objectsChangesQueue.length > 0) {
            this.startRefreshing();
        }

        this.cameraChangeId = this.K3DInstance.on(this.K3DInstance.events.CAMERA_CHANGE, function (control) {
            self.model.set('camera', control);
            self.model.save_changes();
        });

        this.GUIObjectChanges = this.K3DInstance.on(this.K3DInstance.events.OBJECT_CHANGE, function (change) {
            if (self.model._comm_live) {
                objectsList[change.id].save(change.key, change.value);
            }
        });

        this.GUIParametersChanges = this.K3DInstance.on(this.K3DInstance.events.PARAMETERS_CHANGE, function (change) {
            self.model.set(change.key, change.value);
            self.model.save_changes();
        });

        this.voxelsCallback = this.K3DInstance.on(this.K3DInstance.events.VOXELS_CALLBACK, function (param) {
            if (objectsList[param.object.K3DIdentifier]) {
                objectsList[param.object.K3DIdentifier].send({msg_type: 'click_callback', coord: param.coord});
            }
        });
    },

    _setDirectionalLightingIntensity: function () {
        this.K3DInstance.setDirectionalLightingIntensity(this.model.get('lighting'));
    },

    _setTime: function () {
        this.K3DInstance.setTime(this.model.get('time'));
    },

    _setCameraAutoFit: function () {
        this.K3DInstance.setCameraAutoFit(this.model.get('camera_auto_fit'));
    },

    _setGridAutoFit: function () {
        this.K3DInstance.setGridAutoFit(this.model.get('grid_auto_fit'));
    },

    _setGridVisible: function () {
        this.K3DInstance.setGridVisible(this.model.get('grid_visible'));
    },

    _setFpsMeter: function () {
        this.K3DInstance.setFpsMeter(this.model.get('fps_meter'));
    },

    _setScreenshotScale: function () {
        this.K3DInstance.setScreenshotScale(this.model.get('screenshot_scale'));
    },

    _setVoxelPaintColor: function () {
        this.K3DInstance.setVoxelPaint(this.model.get('voxel_paint_color'));
    },

    _setBackgroundColor: function () {
        this.K3DInstance.setClearColor(this.model.get('background_color'));
    },

    _setGrid: function () {
        this.K3DInstance.setGrid(this.model.get('grid'));
    },

    _setMenuVisibility: function () {
        this.K3DInstance.setMenuVisibility(this.model.get('menu_visibility'));
    },

    _setColorMapLegend: function () {
        this.K3DInstance.setColorMapLegend(this.model.get('colorbar_object_id'));
    },

    _setCamera: function () {
        this.K3DInstance.setCamera(this.model.get('camera'));
    },

    _setClippingPlanes: function () {
        this.K3DInstance.setClippingPlanes(this.model.get('clipping_planes'));
    },

    _processObjectsChangesQueue: function (self) {
        var obj;

        if (self.objectsChangesQueue.length === 0) {
            return;
        }

        obj = self.objectsChangesQueue.shift();

        if (obj.operation === 'delete') {
            self.K3DInstance.removeObject(obj.id);
        }

        if (obj.operation === 'insert') {
            self.K3DInstance.load({objects: [objectsList[obj.id].attributes]});
        }

        if (obj.operation === 'update') {
            self.K3DInstance.reload(objectsList[obj.id].attributes);
        }

        if (self.objectsChangesQueue.length > 0) {
            setTimeout(self._processObjectsChangesQueue, 0, self);
        } else {
            self.objectsChangesQueueRun = false;
        }
    },

    _onObjectsListChange: function () {
        var old_object_ids = this.model.previous('object_ids'),
            new_object_ids = this.model.get('object_ids');

        _.difference(old_object_ids, new_object_ids).forEach(function (id) {
            this.objectsChangesQueue.push({id: id, operation: 'delete'});
        }, this);

        _.difference(new_object_ids, old_object_ids).forEach(function (id) {
            this.objectsChangesQueue.push({id: id, operation: 'insert'});
        }, this);

        this.startRefreshing();
    },

    refreshObject: function (obj) {
        if (this.model.get('object_ids').indexOf(obj.get('id')) !== -1) {
            this.objectsChangesQueue.push({id: obj.get('id'), operation: 'update'});
            this.startRefreshing();
        }
    },

    startRefreshing: function () {
        // force setTimeout to avoid freeze on browser in case of heavy load
        if (!this.objectsChangesQueueRun) {
            this.objectsChangesQueueRun = true;
            setTimeout(this._processObjectsChangesQueue, 0, this);
        }
    },

    processPhosphorMessage: function (msg) {
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
        }
    },

    handleEvent: function (event) {
        switch (event.type) {
            case 'contextmenu':
                this.handleContextMenu(event);
                break;
            default:
                widgets.DOMWidgetView.prototype.handleEvent.call(this, event);
                break;
        }
    },

    handleContextMenu: function (event) {
        // Cancel context menu if on renderer:
        if (this.container.contains(event.target)) {
            event.preventDefault();
            event.stopPropagation();
        }
    },

    handleResize: function () {
        this.K3DInstance.resizeHelper();
    }
});

module.exports = {
    PlotModel: PlotModel,
    PlotView: PlotView,
    ObjectModel: ObjectModel,
    ObjectView: ObjectView,
    K3D: K3D,
    ThreeJsProvider: ThreeJsProvider
};
