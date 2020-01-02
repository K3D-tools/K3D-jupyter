'use strict';

function Float16Array(array) {
    var d = new Uint16Array(array);

    d.constructor = Float16Array;

    return d;
}

module.exports = Float16Array;
