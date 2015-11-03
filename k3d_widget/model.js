/* globals define */

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
            },

            _decode: function () {
                var data = this.get('data');

                if (data) {
                    this.set('object', JSON.parse(pako.ungzip(atob(data), {'to': 'string'})));
                }
            }
        })
    };
});
