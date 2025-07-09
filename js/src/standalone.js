const msgpack = require('msgpack-lite');
const fflate = require('fflate');
const TFEdit = require('./transferFunctionEditor');
const serialize = require('./core/lib/helpers/serialize');
const K3D = require('./core/Core');
const ThreeJsProvider = require('./providers/threejs/provider');
const _ = require('./lodash');

const MsgpackCodec = msgpack.createCodec({ preset: true });

const Float16Array = require('./core/lib/helpers/float16Array');

window.Float16Array = Float16Array;

MsgpackCodec.addExtPacker(0x20, Float16Array, (val) => val);
MsgpackCodec.addExtUnpacker(0x20, (val) => Float16Array(val.buffer));

require('katex/dist/katex.min.css');
require('lil-gui/dist/lil-gui.css');

/**
 * Decode msgpack data using the custom codec.
 * @param {Uint8Array|ArrayBuffer} data - The data to decode.
 * @returns {Object} Decoded object.
 */
function msgpackDecode(data) {
    return msgpack.decode(data, { codec: MsgpackCodec });
}

/**
 * Create a K3D instance and load a binary snapshot into the given DOM node.
 * Handles decompression, decoding, and error reporting.
 * @param {ArrayBuffer} data - The binary snapshot data.
 * @param {HTMLElement} targetDOMNode - The DOM node to attach the K3D instance to.
 * @returns {Promise<Object>} Resolves to the K3D instance.
 */
function CreateK3DAndLoadBinarySnapshot(data, targetDOMNode) {
    return new Promise((resolve, reject) => {
        let K3DInstance;
        // Decompress the data using fflate
        fflate.unzlib(new Uint8Array(data), (err, decompressData) => {
            if (!err) {
                data = decompressData;
            }
            // Decode the data using msgpack
            data = msgpackDecode(data);
            try {
                // Create the K3D instance with the decoded plot
                K3DInstance = new K3D(
                    ThreeJsProvider,
                    targetDOMNode,
                    data.plot,
                );
            } catch (e) {
                // Log and reject on error
                console.log(e);
                return reject(e);
            }
            // Set the snapshot and camera, then resolve
            return K3DInstance.setSnapshot(data).then(() => {
                setTimeout(() => {
                    if (data.plot.camera.length > 0) {
                        K3DInstance.setCamera(data.plot.camera);
                        K3DInstance.render();
                    }
                }, 10);
                return resolve(K3DInstance);
            });
        });
    });
}

/**
 * Exported API for standalone K3D usage.
 * @type {Object}
 */
module.exports = {
    K3D,
    msgpackDecode,
    serialize,
    CreateK3DAndLoadBinarySnapshot,
    TransferFunctionEditor: TFEdit.transferFunctionEditor,
    TransferFunctionModel: TFEdit.transferFunctionModel,
    TransferFunctionView: TFEdit.transferFunctionView,
    ThreeJsProvider,
    _,
    version: require('./version').version,
};
