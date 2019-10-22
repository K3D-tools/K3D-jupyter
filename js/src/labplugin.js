// jscs:disable requireVarDeclFirst

'use strict';

require('./deps');

var k3d = require('./index'),
    version = require('./version').version,
    base = require('@jupyter-widgets/base');

module.exports = {
    id: 'jupyter.extensions.k3d',
    requires: [base.IJupyterWidgetRegistry],
    activate: function (app, widgets) {
        widgets.registerWidget({
            name: 'k3d',
            version: version,
            exports: k3d
        });
    },
    autoStart: true
};
