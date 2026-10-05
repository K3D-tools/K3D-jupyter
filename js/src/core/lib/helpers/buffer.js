/**
 * Decodes provided binary string into a ArrayBuffer
 * @method base64ToArrayBuffer
 * @memberof K3D.Helpers
 * @param  {String} base64 BASE64 encoded string
 * @return {ArrayBuffer} from Uint8Array created from decoding base64
 */
function stringToArrayBuffer(binaryString) {
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    let i;

    for (i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    return bytes;
}

function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }

    return window.btoa(binary);
}

/**
 * Decodes provided base64-encoded string into a ArrayBuffer
 * @method base64ToArrayBuffer
 * @memberof K3D.Helpers
 * @param  {String} base64 BASE64 encoded string
 * @return {ArrayBuffer} from Uint8Array created from decoding base64
 */
function base64ToArrayBuffer(base64) {
    return stringToArrayBuffer(window.atob(base64));
}

/**
 * Setup a Float32Array based on input
 * @method colorsToFloat32Array
 * @memberof K3D.Helpers
 * @param  {*} array
 * @return {Float32Array}
 */
function colorsToFloat32Array(array) {
    const colorsArray = new Float32Array(array.length * 3);

    array.forEach((color, i) => {
        colorsArray[i * 3] = ((color >> 16) & 255) / 255;
        colorsArray[i * 3 + 1] = ((color >> 8) & 255) / 255;
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
    const bytes = new Uint8Array(array);
    let i;
    let
        string = '';

    for (i = 0; i < bytes.length; i++) {
        string += String.fromCharCode(bytes[i]);
    }

    return window.btoa(string);
}

/**
 * The format of encoded image bytes, the second part of its image/ MIME type
 * @method imageFormat
 * @memberof K3D.Helpers
 * @param  {Uint8Array} bytes
 * @return {String|null} png, jpeg, gif, webp, bmp - or null for anything else
 */
function imageFormat(bytes) {
    if (!bytes || bytes.length < 12) {
        return null;
    }

    const ascii = (from, to) => String.fromCharCode.apply(null, bytes.subarray(from, to));

    if (bytes[0] === 0x89 && ascii(1, 4) === 'PNG') {
        return 'png';
    }
    if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
        return 'jpeg';
    }
    if (ascii(0, 4) === 'GIF8') {
        return 'gif';
    }
    if (ascii(0, 4) === 'RIFF' && ascii(8, 12) === 'WEBP') {
        return 'webp';
    }
    if (ascii(0, 2) === 'BM') {
        return 'bmp';
    }

    return null;
}

module.exports = {
    imageFormat,
    colorsToFloat32Array,
    bufferToBase64,
    base64ToArrayBuffer,
    arrayBufferToBase64,
    stringToArrayBuffer,
};
