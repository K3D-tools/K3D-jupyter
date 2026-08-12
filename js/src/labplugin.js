// jscs:disable requireVarDeclFirst

const base = require('@jupyter-widgets/base');
const k3d = require('./index');
const { version } = require('./version');

module.exports = {
    id: 'jupyter.extensions.k3d',
    requires: [base.IJupyterWidgetRegistry],
    activate(app, widgets) {
        require('katex/dist/katex.min.css');

        widgets.registerWidget({
            name: 'k3d',
            version,
            exports: k3d,
        });
    },
    autoStart: true,
};
