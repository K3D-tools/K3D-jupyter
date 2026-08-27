require('katex/dist/katex.min.css');

module.exports = {
    K3D: require('./core/Core'),
    TransferFunctionEditor: require('./core/lib/transferFunctionEditorCore'),
    ThreeJsProvider: require('./providers/threejs/provider'),
};
