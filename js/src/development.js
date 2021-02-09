require('es6-promise');

require('style-loader?{attributes:{id: "k3d-katex"}}!css-loader!../node_modules/katex/dist/katex.min.css');
require('style-loader?{attributes:{id: "k3d-dat.gui"}}!css-loader!../node_modules/dat.gui/build/dat.gui.css');

module.exports = {
    K3D: require('./core/Core'),
    TransferFunctionEditor: require('./transferFunctionEditor'),
    ThreeJsProvider: require('./providers/threejs/provider')
};
