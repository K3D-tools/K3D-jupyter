const msgpack = require('msgpack-lite');
const fflate = require('fflate');
const TFEdit = require('./transferFunctionEditor');
const serialize = require('./core/lib/helpers/serialize');
const K3D = require('./core/Core');
const ThreeJsProvider = require('./providers/threejs/provider');

const MsgpackCodec = msgpack.createCodec({ preset: true });

window.Float16Array = require('./core/lib/helpers/float16Array');

MsgpackCodec.addExtPacker(0x20, Float16Array, (val) => val);
MsgpackCodec.addExtUnpacker(0x20, (val) => Float16Array(val.buffer));

require('katex/dist/katex.min.css');
require('lil-gui/dist/lil-gui.css');

function msgpackDecode(data) {
    return msgpack.decode(data, { codec: MsgpackCodec });
}

function CreateK3DAndLoadBinarySnapshot(data, targetDOMNode) {
    return new Promise((resolve, reject) => {
        let K3DInstance;

        fflate.unzlib(new Uint8Array(data), (err, decompressData) => {
            if (!err) {
                data = decompressData;
            }

            data = msgpackDecode(data);

            try {
                K3DInstance = new K3D(
                    ThreeJsProvider,
                    targetDOMNode,
                    data.plot,
                );
            } catch (e) {
                console.log(e);
                return reject(e);
            }

            K3DInstance.setSnapshot(data).then(() => {
                setTimeout(() => {
                    if (data.plot.camera.length > 0) {
                        K3DInstance.setCamera(data.plot.camera);
                        K3DInstance.render();
                    }
                }, 10);

                resolve(K3DInstance);
            });
        });
    });
}

module.exports = {
    K3D,
    msgpackDecode,
    serialize,
    CreateK3DAndLoadBinarySnapshot,
    TransferFunctionEditor: TFEdit.transferFunctionEditor,
    TransferFunctionModel: TFEdit.transferFunctionModel,
    TransferFunctionView: TFEdit.transferFunctionView,
    ThreeJsProvider,
    version: require('./version').version,
};
