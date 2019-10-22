'use strict';

var pako = require('pako'),
    Float16Array = require('./float16Array'),
    typesToArray = {
        int8: Int8Array,
        int16: Int16Array,
        int32: Int32Array,
        uint8: Uint8Array,
        uint16: Uint16Array,
        uint32: Uint32Array,
        float16: Float16Array,
        float32: Float32Array,
        float64: Float64Array
    };

function isNumeric(n) {
    return !isNaN(parseFloat(n)) && isFinite(n);
}

function deserializeArray(obj) {
    var buffer;

    if (typeof (obj.buffer) !== 'undefined') {
        console.log('K3D: Receive: ' + obj.buffer.byteLength + ' bytes');
        return {
            data: new typesToArray[obj.dtype](obj.buffer.buffer),
            shape: obj.shape
        };
    } else if (typeof (obj.compressed_buffer) !== 'undefined') {
        buffer = new typesToArray[obj.dtype](pako.inflate(obj.compressed_buffer.buffer).buffer);

        console.log('K3D: Receive: ' + buffer.byteLength + ' bytes compressed to ' +
                    obj.compressed_buffer.byteLength + ' bytes');

        return {
            data: buffer,
            shape: obj.shape
        };
    }
}

function serializeArray(obj) {
    return {
        dtype: _.invert(typesToArray)[obj.data.constructor],
        compressed_buffer: pako.deflate(obj.data.buffer, {level: 9}),
        shape: obj.shape
    };
}

function deserialize(obj, manager) {
    if (obj == null) {
        return null;
    } else if (typeof (obj) === 'string' || typeof(obj) === 'boolean') {
        return obj;
    } else if (_.isNumber(obj)) { // plain number
        return obj;
    } else if (typeof (obj.shape) !== 'undefined') {
        // plain data
        return deserializeArray(obj);
    } else if (Array.isArray(obj)) {
        return obj.reduce(function (p, v) {
            p.push(deserialize(v, manager));

            return p;
        }, []);
    } else {
        // time series or dict
        var timeSeries = true;
        var deserializedObj = Object.keys(obj).reduce(function (p, k) {
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
}

function serialize(obj) {
    if (_.isNumber(obj)) {
        return obj;
    } else if (typeof (obj) === 'string' || typeof(obj) === 'boolean') {
        return obj;
    }

    if (obj !== null) {
        if (typeof (obj.data) !== 'undefined' && typeof (obj.shape) !== 'undefined' && typeof (obj.data) !== 'undefined') {
            // plain data
            return serializeArray(obj);
        } else if (Array.isArray(obj)) {
            return obj.reduce(function (p, v) {
                p.push(serialize(v));

                return p;
            }, []);
        } else {
            // time series or dict
            return Object.keys(obj).reduce(function (p, k) {
                p[k] = serialize(obj[k]);

                return p;
            }, {});
        }
    } else {
        return null;
    }
}

module.exports = {
    deserialize: deserialize,
    serialize: serialize
};
