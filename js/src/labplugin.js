var k3d = require('./index');
var EXTENSION_SPEC_VERSION = require('./version').EXTENSION_SPEC_VERSION;
var base = require('@jupyter-widgets/base');

module.exports = {
    id: 'k3d',
    requires: [base.IJupyterWidgetRegistry],
    activate: function(app, widgets) {
        widgets.registerWidget({
            name: 'k3d',
            version: EXTENSION_SPEC_VERSION,
            exports: k3d
        });
    },
    autoStart: true
};
