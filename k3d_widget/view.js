/* globals define, $ */

requirejs.config({
    paths: {
        'k3d': '../nbextensions/k3d_widget/lib/k3d'
    },
    shim: {
        'k3d/k3d.min': {
            deps: ['k3d/k3d.deps.min']
        },
        'k3d/providers/k3d.threejs.min': {
            deps: ['k3d/k3d.min', 'k3d/providers/k3d.threejs.deps.min']
        }
    }
});

define(['nbextensions/widgets/widgets/js/widget', 'k3d/providers/k3d.threejs.min'], function (widget) {

    return {
        K3DView: widget.DOMWidgetView.extend({
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
                var renderer = new THREE.WebGLRenderer({
                        antialias: this.parameters.antialias,
                    }),
                    self = this;

                this.K3D = K3D.Core(K3D.Providers.ThreeJS, this.container, {
                    renderer: renderer
                }, function (newInstance) {
                    self.K3D = newInstance;
                });

                renderer.setClearColor(this.parameters.backgroundColor);
                this._setCameraAutoFit();
                this._setVoxelPaintColor();
            },

            _setCameraAutoFit: function () {
                this.K3D.setCameraAutoFit(this.model.get('camera_auto_fit'));
            },

            _setVoxelPaintColor: function () {
                this.K3D.parameters.voxelPaintColor = this.model.get('voxel_paint_color');
                console.log(this.K3D.parameters.voxelPaintColor);
            },

            _onObjectChange: function (model, object) {
                if (object.type) {
                    return this._add(object);
                }

                (object.attr ? this._update : this._remove).call(this, object);
            },

            _add: function (object) {
                K3D.Loader(this.K3D, {objects: [object]});
            },

            _update: function (object) {
                var patch = [{
                    'op': 'replace',
                    'path': object.attr.path,
                    'value': object.attr.value
                }];

                K3D.applyPatchObject(this.K3D, object.id, patch);
            },

            _remove: function (object) {
                K3D.removeObject(this.K3D, object.id);
            },

            _fetchData: function (id) {
                this.send(K3D.getPatchObject(this.K3D, id));
            }
        })
    };
});
