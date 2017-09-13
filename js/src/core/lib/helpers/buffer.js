'use strict';

/**
 * Decodes provided base64-encoded string into a ArrayBuffer
 * @method base64ToArrayBuffer
 * @memberof K3D.Helpers
 * @param  {String} base64 BASE64 encoded string
 * @return {ArrayBuffer} from Uint8Array created from decoding base64
 */
function base64ToArrayBuffer(base64) {
    var binary_string = window.atob(base64),
        len = binary_string.length,
        bytes = new Uint8Array(len),
        i;

    for (i = 0; i < len; i++) {
        bytes[i] = binary_string.charCodeAt(i);
    }

    return bytes.buffer;
}

/**
 * Setup a Float32Array based on input
 * @method colorsToFloat32Array
 * @memberof K3D.Helpers
 * @param  {*} array
 * @return {Float32Array}
 */
function colorsToFloat32Array(array) {
    var colorsArray;

    colorsArray = new Float32Array(array.length * 3);

    array.forEach(function (color, i) {
        colorsArray[i * 3] = (color >> 16 & 255) / 255;
        colorsArray[i * 3 + 1] = (color >> 8 & 255) / 255;
        colorsArray[i * 3 + 2] = (color & 255) / 255;
    });

    return colorsArray;
}

/**
 * convert buffer to base64
 * @method bufferToBase64
 * @memberof K3D.Helpers
 * @param  {Buffer} array
 * @return {String}
 */
function bufferToBase64(array) {
    var bytes = new Uint8Array(array), i, string = '';

    for (i = 0; i < bytes.length; i++) {
        string += String.fromCharCode(bytes[i]);
    }

    return window.btoa(string);
}

module.exports = {
    colorsToFloat32Array: colorsToFloat32Array,
    bufferToBase64: bufferToBase64,
    base64ToArrayBuffer: base64ToArrayBuffer
};
