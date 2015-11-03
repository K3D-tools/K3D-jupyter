/* globals define, $ */

requirejs.config({
    paths: {
        'k3d': '../nbextensions/k3d_widget/lib/k3d'
    },
    shim: {
        'k3d/providers/k3d.threejs.min': {
            deps: ['k3d/k3d.min', 'k3d/providers/k3d.threejs.deps.min']
        }
    }
});

define(['nbextensions/widgets/widgets/js/widget', 'k3d/providers/k3d.threejs.min'], function (widget) {

    return {
        K3DView: widget.DOMWidgetView.extend({
            render: function () {
                this.canvas = $('<div />').css({'height': this.model.get('height')}).appendTo(this.$el).get(0);
                this.on('displayed', this._init, this);
                this.model.on('change:object', this._add, this);
            },

            _init: function () {
                this.K3D = K3D.Core(K3D.Providers.ThreeJS, this.canvas, {
                    renderer: new THREE.WebGLRenderer()
                });
            },

            _add: function () {
                K3D.Loader(this.K3D, {objects: [this.model.get('object')]});
            }
        })
    };
});
