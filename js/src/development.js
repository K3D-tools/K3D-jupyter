require('es6-promise');

//TODO
// require('style-loader?{attributes:{id: "k3d-katex"}}!css-loader!katex/dist/katex.min.css');
// require('style-loader?{attributes:{id: "k3d-dat.gui"}}!css-loader!dat.gui/build/dat.gui.css');

require('katex/dist/katex.min.css');
require('dat.gui/build/dat.gui.css');


module.exports = {
    K3D: require('./core/Core'),
    TransferFunctionEditor: require('./transferFunctionEditor'),
    ThreeJsProvider: require('./providers/threejs/provider'),
};
