'use strict';

function createSVGGradient(svg, id, colormap, opacity, horizontal) {
    var svgNS = svg.namespaceURI,
        data,
        grad = document.createElementNS(svgNS, 'linearGradient'),
        min, max;

    if (svg.getElementById(id)) {
        svg.getElementById(id).remove();
    }

    grad.setAttribute('id', id);

    if (horizontal) {
        grad.setAttribute('x1', '0');
        grad.setAttribute('x2', '1');
        grad.setAttribute('y1', '0');
        grad.setAttribute('y2', '0');
    } else {
        grad.setAttribute('x1', '0');
        grad.setAttribute('x2', '0');
        grad.setAttribute('y1', '1');
        grad.setAttribute('y2', '0');
    }

    data = mergeColorMapWithOpacity(colormap, opacity);

    min = data[0];
    max = data[data.length - 5];

    for (var i = 0; i < data.length; i += 5) {
        var stop = document.createElementNS(svgNS, 'stop'),
            segment = data.slice(i, i + 5);

        stop.setAttribute('offset', ((segment[0] - min) / (max - min)).toString(10));
        stop.setAttribute('stop-color', 'rgb(' +
            toColor(segment[1]) + ',' +
            toColor(segment[2]) + ',' +
            toColor(segment[3]) + ')');
        stop.setAttribute('stop-opacity', segment[4]);
        grad.appendChild(stop);
    }

    var defs = svg.querySelector('defs') ||
        svg.insertBefore(document.createElementNS(svgNS, 'defs'), svg.firstChild);

    return defs.appendChild(grad);
}

function mergeColorMapWithOpacity(colormap, opacity) {
    var merged = {}, sortedKeys, i;

    function findNeighbors(key, property) {
        var startKeyIndex = sortedKeys.indexOf(key), leftIndex = startKeyIndex, rightIndex = startKeyIndex;

        while (typeof (merged[sortedKeys[leftIndex]][property]) === 'undefined' && leftIndex > 0) {
            leftIndex--;
        }

        while (typeof (merged[sortedKeys[rightIndex]][property]) === 'undefined' && rightIndex < sortedKeys.length - 1) {
            rightIndex++;
        }

        return [sortedKeys[leftIndex], sortedKeys[rightIndex]];
    }

    function interpolate(key, property) {
        var neighbors = findNeighbors(key, property),
            t = (key - neighbors[0]) / (neighbors[1] - neighbors[0]),
            a, b;

        if (typeof (merged[neighbors[0]][property]) !== 'undefined') {
            a = merged[neighbors[0]][property];
        } else {
            a = 0.0;
        }

        if (typeof (merged[neighbors[1]][property]) !== 'undefined') {
            b = merged[neighbors[1]][property];
        } else {
            b = 1.0;
        }

        return a * (1 - t) + b * t;
    }

    for (i = 0; i < colormap.length; i += 4) {
        merged[colormap[i]] = {
            r: colormap[i + 1],
            g: colormap[i + 2],
            b: colormap[i + 3]
        };
    }

    if (!opacity) {
        sortedKeys = Object.keys(merged).map(parseFloat).sort(function (a, b) {
            return a - b;
        });
        opacity = [sortedKeys[0], 1.0, sortedKeys[sortedKeys.length - 1], 1.0];
    }

    for (i = 0; i < opacity.length; i += 2) {
        if (!merged[opacity[i]]) {
            merged[opacity[i]] = {};
        }

        merged[opacity[i]].a = opacity[i + 1];
    }

    sortedKeys = Object.keys(merged).map(parseFloat).sort(function (a, b) {
        return a - b;
    });

    sortedKeys.forEach(function (key) {
        if (typeof (merged[key].a) === 'undefined') {
            merged[key].a = interpolate(key, 'a');
        }

        if (typeof (merged[key].r) === 'undefined') {
            merged[key].r = interpolate(key, 'r');
            merged[key].g = interpolate(key, 'g');
            merged[key].b = interpolate(key, 'b');
        }
    });

    return sortedKeys.reduce(function (prev, key) {
        return prev.concat([parseFloat(key), merged[key].r, merged[key].g, merged[key].b, merged[key].a]);
    }, []);
}


function toColor(val) {
    if (val > 1.0) {
        val = 1.0;
    }

    if (val < 0.0) {
        val = 0.0;
    }

    return Math.round((val * 255));
}

function createCanvasGradient(colorMap, size, opacityFunction) {
    var canvas = document.createElement('canvas'),
        ctx = canvas.getContext('2d'),
        grd,
        segment,
        merged,
        i,
        min, max;

    merged = mergeColorMapWithOpacity(colorMap, opacityFunction);
    canvas.height = 1;
    canvas.width = size;

    grd = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);

    min = merged[0];
    max = merged[merged.length - 5];

    for (i = 0; i < merged.length / 5; i++) {
        segment = merged.slice(i * 5, i * 5 + 5);

        grd.addColorStop((segment[0] - min) / (max - min),
            'rgba(' +
            toColor(segment[1]) + ', ' +
            toColor(segment[2]) + ', ' +
            toColor(segment[3]) + ', ' +
            segment[4] +
            ')');
    }

    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    return canvas;
}

function createCanvasColorList(values) {
    var canvas = document.createElement('canvas'),
        ctx = canvas.getContext('2d'),
        i;

    canvas.height = 1;
    canvas.width = 256;

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    for (i = 0; i < values.length; i++) {
        imageData.data[i * 4 + 3] = 0xff;
        imageData.data[i * 4] = (values[i] >> 16) & 0xff;
        imageData.data[i * 4 + 1] = (values[i] >> 8) & 0xff;
        imageData.data[i * 4 + 2] = values[i] & 0xff;
    }

    ctx.putImageData(imageData, 0, 0);

    return canvas;
}

module.exports = {
    createCanvasGradient: createCanvasGradient,
    createCanvasColorList: createCanvasColorList,
    createSVGGradient: createSVGGradient,
    mergeColorMapWithOpacity: mergeColorMapWithOpacity
};

