'use strict';

var widgets = require('jupyter-js-widgets'),
    _ = require('lodash'),
    pako = require('pako'),
    jsonpatch = require('fast-json-patch'),
    K3D = require('./core/Core'),
    ThreeJsProvider = require('./providers/threejs/provider'),
    K3DModel,
    K3DView,
    semverRange = '~' + require('../package.json').version;

require('es6-promise');

// Custom Model. Custom widgets models must at least provide default values
// for model attributes, including
//
//  - `_view_name`
//  - `_view_module`
//  - `_view_module_version`
//
//  - `_model_name`
//  - `_model_module`
//  - `_model_module_version`
//
//  when different from the base class.

// When serialiazing the entire widget state for embedding, only values that
// differ from the defaults will be specified.

function createPatchFromObject(object) {
    return [{
        'op': 'replace',
        'path': object.attr.path,
        'value': object.attr.value
    }];
}

K3DModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(_.result({}, 'widgets.DOMWidgetModel.prototype.defaults'), {
        _model_name: 'K3D',
        _view_name: 'K3D',
        _model_module: 'K3D',
        _view_module: 'K3D',
        _model_module_version: semverRange,
        _view_module_version: semverRange,
        value: 'K3D'
    }),

    updateHistory: function (obj) {
        switch (obj.k3dOperation) {
            case 'Insert':
                this.objectsList[obj.id] = _.omit(obj, 'k3dOperation');
                break;
            case 'Update':
                jsonpatch.apply(this.objectsList[obj.id], createPatchFromObject(obj));
                break;
            case 'Delete':
                delete this.objectsList[obj.id];
        }
    },

    initialize: function () {
        widgets.WidgetModel.prototype.initialize.apply(this, arguments);
        this.on('change:data', this._decode, this);
        this.on('msg:custom', this._fetchData, this);
        this.objectsList = {};
    },

    _decode: function () {
        var obj, data = this.get('data');

        if (data) {
            obj = JSON.parse(pako.ungzip(atob(data), {'to': 'string'}));

            if (typeof (obj.k3dOperation) !== 'undefined') {
                this.updateHistory(obj);
                this.set('object', obj);
            }
        }
    },

    _fetchData: function (id) {
        this.trigger('fetchData', id);
    }
});

// Custom View. Renders the widget model.
K3DView = widgets.DOMWidgetView.extend({
    render: function () {
        var container = $('<div />').css('position', 'relative');

        this.container = container.css({'height': this.model.get('height')}).appendTo(this.$el).get(0);
        this.on('displayed', this._init, this);

        this.model.on('change:camera_auto_fit', this._setCameraAutoFit, this);
        this.model.on('change:grid_auto_fit', this._setGridAutoFit, this);
        this.model.on('change:voxel_paint_color', this._setVoxelPaintColor, this);
        this.model.on('change:background_color', this._setBackgroundColor, this);
        this.model.on('change:grid', this._setGrid, this);
        this.model.on('change:camera', this._setCamera, this);

        this.model.on('change:object', this._onObjectChange, this);
        this.model.on('fetchData', this._fetchData, this);
    },

    _init: function () {
        var self = this;

        try {
            this.K3DInstance = new K3D(ThreeJsProvider, this.container, {
                antialias: this.model.get('antialias'),
                ObjectsListJson: this.model.objectsList
            });
        } catch (e) {
            return;
        }

        this.objectsChangesQueue = [];
        this.objectsChangesQueueRun = false;

        this.K3DInstance.setClearColor(this.model.get('background_color'));

        this._setCameraAutoFit();
        this._setGridAutoFit();
        this._setVoxelPaintColor();

        Object.keys(this.model.objectsList).forEach(function (key) {
            var obj = _.extend({}, this.model.objectsList[key], {'k3dOperation': 'Insert'});

            this._onObjectChange(this.model, obj);
        }, this);

        this.K3DInstance.on(this.K3DInstance.events.CAMERA_CHANGE, function (control) {
            self.model.set('camera', control);
            self.model.save_changes();
        });
    },

    _setCameraAutoFit: function () {
        this.K3DInstance.setCameraAutoFit(this.model.get('camera_auto_fit'));
    },

    _setGridAutoFit: function () {
        this.K3DInstance.setGridAutoFit(this.model.get('grid_auto_fit'));
    },

    _setVoxelPaintColor: function () {
        this.K3DInstance.parameters.voxelPaintColor = this.model.get('voxel_paint_color');
    },

    _setBackgroundColor: function () {
        this.K3DInstance.setClearColor(this.model.get('background_color'));
    },

    _setGrid: function () {
        this.K3DInstance.setGrid(this.model.get('grid'));
    },

    _setCamera: function () {
        this.K3DInstance.setCamera(this.model.get('camera'));
    },

    _processObjectsChangesQueue: function (self) {
        var object = self.objectsChangesQueue.shift();

        switch (object.k3dOperation) {
            case 'Insert':
                self.K3DInstance.load({objects: [_.omit(object, 'k3dOperation')]}, false);
                break;
            case 'Update':
                object = self.model.objectsList[object.id];
                self.K3DInstance.reload(object, false);
                break;
            case 'Delete':
                self.K3DInstance.removeObject(object.id, false);
        }

        if (self.objectsChangesQueue.length > 0) {
            setTimeout(self._processObjectsChangesQueue, 0, self);
        } else {
            self.objectsChangesQueueRun = false;
        }
    },

    _onObjectChange: function (model, object) {
        this.objectsChangesQueue.push(object);

        // force setTimeout to avoid freeze on browser in case of heavy load
        if (!this.objectsChangesQueueRun) {
            this.objectsChangesQueueRun = true;
            setTimeout(this._processObjectsChangesQueue, 0, this);
        }
    },

    _fetchData: function (id) {
        var currentObjectJson = this.K3DInstance.getObjectJson(id);

        if (currentObjectJson !== null) {
            this.send({
                type: 'object',
                id: id,
                'data': jsonpatch.compare(this.model.objectsList[id], currentObjectJson)
            });
        }
    }
});

module.exports = {
    K3DModel: K3DModel,
    K3DView: K3DView
};
