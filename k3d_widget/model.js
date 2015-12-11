/* globals define, requirejs, atob */

requirejs.config({
    paths: {
        'pako': '../nbextensions/k3d_widget/lib/pako/dist'
    }
});

define(['nbextensions/widgets/widgets/js/widget', 'pako/pako_inflate.min'], function(widget, pako) {
    'use strict';

    return {
        K3DModel: widget.WidgetModel.extend({
            initialize: function () {
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
