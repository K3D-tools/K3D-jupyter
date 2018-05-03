var k3d = require('./index');
var base = require('@jupyter-widgets/base');

module.exports = {
    id: 'k3d',
    requires: [base.IJupyterWidgetRegistry],
    activate: function(app, widgets) {
        widgets.registerWidget({
            name: 'k3d',
            version: k3d.version,
            exports: k3d
        });
    },
    autoStart: true
};
