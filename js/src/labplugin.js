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
        require('style-loader?{attrs:{id: "k3d-katex"}}!css-loader!../node_modules/katex/dist/katex.min.css');
        require('style-loader?{attrs:{id: "k3d-dat.gui"}}!css-loader!../node_modules/dat.gui/build/dat.gui.css');

        widgets.registerWidget({
            name: 'k3d',
            version: version,
            exports: k3d
        });
    },
    autoStart: true
};
