const fflate = require('fflate');
const _ = require('../../../lodash');
const Float16Array = require('./float16Array');
const msgpack = require('msgpack-lite');
const buffer = require('./buffer');

const typesToArray = {
    int8: Int8Array,
    int16: Int16Array,
    int32: Int32Array,
    uint8: Uint8Array,
    uint16: Uint16Array,
    uint32: Uint32Array,
    float16: Float16Array,
    float32: Float32Array,
    float64: Float64Array,
};
const MsgpackCodec = msgpack.createCodec({ preset: true });
MsgpackCodec.addExtPacker(0x20, Float16Array, (val) => val);
MsgpackCodec.addExtUnpacker(0x20, (val) => Float16Array(val.buffer));


function isNumeric(n) {
    return !Number.isNaN(parseFloat(n)) && Number.isFinite(parseFloat(n));
}

function deserializeArray(obj) {
    let data;

    if (typeof (obj.data) !== 'undefined') {
        return {
            data: new typesToArray[obj.dtype](obj.data.buffer),
            shape: obj.shape,
        };
    }
    if (typeof (obj.compressed_data) !== 'undefined') {
        data = new typesToArray[obj.dtype](fflate.unzlibSync(new Uint8Array(obj.compressed_data.buffer)).buffer);

        console.log(`K3D: Receive: ${data.byteLength} bytes compressed to ${
            obj.compressed_data.byteLength} bytes`);

        return {
            data: data,
            shape: obj.shape,
        };
    }
    return obj;
}

function serializeArray(obj) {
    if (obj.compression_level && obj.compression_level > 0) {
        return {
            dtype: _.invert(typesToArray)[obj.data.constructor],
            compressed_data: fflate.zlibSync(obj.data.buffer, { level: obj.compression_level }),
            shape: obj.shape,
        };
    }
    return {
        dtype: _.invert(typesToArray)[obj.data.constructor],
        data: obj.data,
        shape: obj.shape,
    };
}

function deserialize(obj, manager) {
    if (obj == null) {
        return null;
    }
    if (typeof (obj) === 'string' && obj.substring(0, 7) === 'base64_') {
        obj = msgpack.decode(buffer.base64ToArrayBuffer(obj.substring(7)), { codec: MsgpackCodec });
    }
    if (typeof (obj) === 'string' || typeof (obj) === 'boolean') {
        return obj;
    }
    if (_.isNumber(obj)) { // plain number
        return obj;
    }
    if (typeof (obj.shape) !== 'undefined') {
        // plain data
        return deserializeArray(obj);
    }
    if (Array.isArray(obj)) {
        return obj.reduce((p, v) => {
            p.push(deserialize(v, manager));

            return p;
        }, []);
    }
    // time series or dict
    let timeSeries = true;
    const deserializedObj = Object.keys(obj).reduce((p, k) => {
        if (!isNumeric(k)) {
            timeSeries = false;
        }

        p[k] = deserialize(obj[k], manager);

        return p;
    }, {});

    if (timeSeries) {
        deserializedObj.timeSeries = true;
    }

    return deserializedObj;
}

function serialize(obj) {
    if (_.isNumber(obj)) {
        return obj;
    }
    if (typeof (obj) === 'string' || typeof (obj) === 'boolean') {
        return obj;
    }

    if (obj !== null) {
        if (typeof (obj.data) !== 'undefined' && typeof (obj.shape) !== 'undefined') {
            // plain data
            return serializeArray(obj);
        }
        if (Array.isArray(obj)) {
            return obj.reduce((p, v) => {
                p.push(serialize(v));

                return p;
            }, []);
        }
        // time series or dict
        return Object.keys(obj).reduce((p, k) => {
            p[k] = serialize(obj[k]);

            return p;
        }, {});
    }
    return null;
}

module.exports = {
    deserialize,
    serialize,
};
