'use strict';

var _ = require('lodash'),
    pako = require('pako'),
    ipyDataWidgets = require('jupyter-dataserializers').data_union_array_serialization,
    typesToArray = {
        int8: Int8Array,
        int16: Int16Array,
        int32: Int32Array,
        uint8: Uint8Array,
        uint16: Uint16Array,
        uint32: Uint32Array,
        float32: Float32Array,
        float64: Float64Array
    };

// Inpspiration from https://github.com/maartenbreddels/ipyvolume/blob/master/js/src/serialize.js
// and https://github.com/jupyter-widgets/ipywidgets/blob/master/jupyter-widgets-base/test/src/dummy-manager.ts

function deserialize_array_or_json(obj, manager) {
    var buffer;

    if (obj == null) {
        return null;
    }

    if (typeof(obj) === 'string') {
        return ipyDataWidgets.deserialize(obj, manager);
    }

    if (_.isNumber(obj)) { // plain number
        return obj;
    } else { // should be an array of buffer+dtype+shape
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
        } else {
            return null;
        }
    }
}

function serialize_array_or_json(obj) {
    if (_.isNumber(obj)) {
        return obj;
    }

    if (obj !== null) {
        return {
            dtype: _.invert(typesToArray)[obj.buffer.constructor],
            buffer: obj.buffer,
            shape: obj.shape
        };
    } else {
        return null;
    }
}

module.exports = {
    array_or_json: {deserialize: deserialize_array_or_json, serialize: serialize_array_or_json}
};
