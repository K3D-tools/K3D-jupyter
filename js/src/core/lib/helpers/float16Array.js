'use strict';

function Float16Array(array) {
    return new Uint16Array(array);
}

Float16Array.prototype = Uint16Array.prototype;
Float16Array.prototype.constructor = Float16Array;

module.exports = Float16Array;
