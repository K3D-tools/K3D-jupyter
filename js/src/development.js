require('es6-promise');
require('katex/dist/katex.min.css');
require('dat.gui/build/dat.gui.css');

module.exports = {
    K3D: require('./core/Core'),
    TransferFunctionEditor: require('./transferFunctionEditor'),
    ThreeJsProvider: require('./providers/threejs/provider'),
};
