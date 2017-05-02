'use strict';

var widgets = require('jupyter-js-widgets'),
    _ = require('lodash'),
    pako = require('pako'),
    K3D = require('./core/Core'),
    ThreeJsProvider = require('./providers/threejs/provider'),
    K3DModel,
    K3DView;

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
K3DModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(_.result({}, 'widgets.DOMWidgetModel.prototype.defaults'), {
        _model_name: 'K3D',
        _view_name: 'K3D',
        _model_module: 'K3D',
        _view_module: 'K3D',
        _model_module_version: '2.0.0',
        _view_module_version: '2.0.0',
        value: 'K3D'
    }),

    initialize: function () {
        widgets.WidgetModel.prototype.initialize.apply(this, arguments);
        this.on('change:data', this._decode, this);
        this.on('msg:custom', this._fetchData, this);
    },

    _decode: function () {
        var data = this.get('data');

        if (data) {
            this.set('object', JSON.parse(pako.ungzip(atob(data), {'to': 'string'})));
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

        this.parameters = this.model.get('parameters');
        this.container = container.css({'height': this.parameters.height}).appendTo(this.$el).get(0);
        this.on('displayed', this._init, this);

        this.model.on('change:camera_auto_fit', this._setCameraAutoFit, this);
        this.model.on('change:object', this._onObjectChange, this);
        this.model.on('change:voxel_paint_color', this._setVoxelPaintColor, this);
        this.model.on('fetchData', this._fetchData, this);
    },

    _init: function () {
        this.K3DInstance = new K3D(ThreeJsProvider, this.container);
        this.objectsChangesQueue = [];
        this.objectsChangesQueueRun = false;

        this.K3DInstance.setClearColor(this.parameters.backgroundColor);
        this._setCameraAutoFit();
        this._setVoxelPaintColor();
    },

    _setCameraAutoFit: function () {
        this.K3DInstance.setCameraAutoFit(this.model.get('camera_auto_fit'));
    },

    _setVoxelPaintColor: function () {
        this.K3DInstance.parameters.voxelPaintColor = this.model.get('voxel_paint_color');
    },

    _processObjectsChangesQueue: function (self) {
        var object = self.objectsChangesQueue.shift();

        if (object.type) {
            self._add(object);
        } else {
            (object.attr ? self._update : self._remove).call(self, object);
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

    _add: function (object) {
        this.K3DInstance.load({objects: [object]});
    },

    _update: function (object) {
        var patch = [{
            'op': 'replace',
            'path': object.attr.path,
            'value': object.attr.value
        }];

        this.K3DInstance.applyPatchObject(object.id, patch);
    },

    _remove: function (object) {
        this.K3DInstance.removeObject(object.id);
    },

    _fetchData: function (id) {
        this.send(this.K3DInstance.getPatchObject(id));
    }
});

module.exports = {
    K3DModel: K3DModel,
    K3DView: K3DView
};
