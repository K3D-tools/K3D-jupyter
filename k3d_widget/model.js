/* globals define, requirejs, atob */

requirejs.config({
    paths: {
        'pako': '../nbextensions/k3d_widget/lib/pako/dist',
        'k3d': '../nbextensions/k3d_widget/'
    }
});

define(['jupyter-js-widgets', 'pako/pako_inflate.min'], function(widget, pako) {
    'use strict';

    return {
        K3DModel: widget.WidgetModel.extend({
            initialize: function () {
                widget.WidgetView.prototype.initialize.apply(this, arguments);
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
        })
    };
});
