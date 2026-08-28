const { encode: msgpackEncode, decode: msgpackDecode, ExtensionCodec } = require('@msgpack/msgpack');
const Float16Array = require('./float16Array');

// Typed arrays travel as extension types. 0x11..0x1D are the assignments msgpack-lite used to
// provide as its "preset" codec, and 0x20 is K3D's own for the Float16Array stand-in.
// MessagePack leaves 0..127 to applications, so none of these is standardised - they are a
// contract with the .k3d files and exported HTML already in circulation, and cannot change.
const extensionCodec = new ExtensionCodec();

function bytesOf(view) {
    return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

// A decoded payload is a view into the whole message, at an arbitrary offset, so it has to be
// copied before being reinterpreted as a typed array.
function typedFrom(Type, data) {
    const copy = data.slice();
    return new Type(copy.buffer, 0, copy.byteLength / Type.BYTES_PER_ELEMENT);
}

function register(type, Constructor, claims) {
    extensionCodec.register({
        type,
        encode: (input) => (claims(input) ? bytesOf(input) : null),
        decode: (data) => typedFrom(Constructor, data),
    });
}

const isPlain = (Constructor) => (input) => input instanceof Constructor;

register(0x11, Int8Array, isPlain(Int8Array));
register(0x12, Uint8Array, isPlain(Uint8Array));
register(0x13, Int16Array, isPlain(Int16Array));
// The stand-in below is a Uint16Array, so it has to be excluded here explicitly: the codec
// dispatches by type code, not by registration order, and 0x14 is tried before 0x20.
register(0x14, Uint16Array, (input) => input instanceof Uint16Array
    && input.constructor !== Float16Array);
register(0x15, Int32Array, isPlain(Int32Array));
register(0x16, Uint32Array, isPlain(Uint32Array));
register(0x17, Float32Array, isPlain(Float32Array));
register(0x18, Float64Array, isPlain(Float64Array));
register(0x19, Uint8ClampedArray, isPlain(Uint8ClampedArray));

extensionCodec.register({
    type: 0x1A,
    encode: (input) => (input instanceof ArrayBuffer ? new Uint8Array(input) : null),
    decode: (data) => data.slice().buffer,
});

extensionCodec.register({
    type: 0x1D,
    encode: (input) => (input instanceof DataView ? bytesOf(input) : null),
    decode: (data) => new DataView(data.slice().buffer),
});

extensionCodec.register({
    type: 0x20,
    encode: (input) => ((input && input.constructor === Float16Array) ? bytesOf(input) : null),
    decode: (data) => Float16Array(data.slice().buffer),
});

function encode(data) {
    return msgpackEncode(data, { extensionCodec });
}

function decode(data) {
    return msgpackDecode(
        data instanceof ArrayBuffer ? new Uint8Array(data) : data,
        { extensionCodec },
    );
}

module.exports = {
    encode,
    decode,
    // K3D.Core republishes this as K3DInstance.MsgpackCodec, so it is reachable from user code.
    codec: extensionCodec,
};
