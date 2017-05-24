'use strict';

function toColor(val) {
    if (val > 1.0) {
        val = 1.0;
    }

    if (val < 0.0) {
        val = 0.0;
    }

    return Math.round((val * 255));
}

module.exports = function (colorMap, size) {
    var canvas = document.createElement('canvas');
    canvas.height = 1;
    canvas.width = size;

    var ctx = canvas.getContext('2d');

    var grd = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);

    for (var i = 0; i < colorMap.length / 4; i++) {
        var segment = colorMap.slice(i * 4, i * 4 + 4);

        grd.addColorStop(segment[0],
            'rgb(' + toColor(segment[1]) + ', ' + toColor(segment[2]) + ', ' + toColor(segment[3]) + ')');
    }

    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    return canvas;
};