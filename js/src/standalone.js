const TFEdit = require('./transferFunctionEditor');
const K3D = require('./core/Core');
const ThreeJsProvider = require('./providers/threejs/provider');
const msgpack = require('msgpack-lite');
const pako = require('pako');
const MsgpackCodec = msgpack.createCodec({ preset: true });

require('style-loader?{attributes:{id: "k3d-katex"}}!css-loader!../node_modules/katex/dist/katex.min.css');
require('style-loader?{attributes:{id: "k3d-dat.gui"}}!css-loader!../node_modules/dat.gui/build/dat.gui.css');

function CreateK3DAndLoadBinarySnapshot(data, targetDOMNode) {
    let K3DInstance;

    data = pako.inflate(data);
    data = msgpack.decode(data, { codec: MsgpackCodec });

    try {
        K3DInstance = new K3D(
            ThreeJsProvider,
            targetDOMNode,
            data.plot
        );
    } catch (e) {
        console.log(e);
        return;
    }

    return K3DInstance.setSnapshot(data).then(function () {
        setTimeout(function () {
            if (data.plot.camera.length > 0) {
                K3DInstance.setCamera(data.plot.camera);
                K3DInstance.render();
            }
        }, 10);

        return K3DInstance;
    });
}

module.exports = {
    K3D: K3D,
    CreateK3DAndLoadBinarySnapshot: CreateK3DAndLoadBinarySnapshot,
    TransferFunctionEditor: TFEdit.transferFunctionEditor,
    TransferFunctionModel: TFEdit.transferFunctionModel,
    TransferFunctionView: TFEdit.transferFunctionView,
    ThreeJsProvider: ThreeJsProvider,
    version: require('./version').version
};
