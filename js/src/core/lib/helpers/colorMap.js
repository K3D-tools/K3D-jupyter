function toColor(val) {
    if (val > 1.0) {
        val = 1.0;
    }

    if (val < 0.0) {
        val = 0.0;
    }

    return Math.round((val * 255));
}

function mergeColorMapWithOpacity(colormap, opacity, dim) {
    const merged = {};
    let sortedKeys;
    let i;

    function findNeighbors(key, property) {
        const startKeyIndex = sortedKeys.indexOf(key);
        let leftIndex = startKeyIndex;
        let
            rightIndex = startKeyIndex;

        while (typeof (merged[sortedKeys[leftIndex]][property]) === 'undefined' && leftIndex > 0) {
            leftIndex -= 1;
        }

        while (typeof (merged[sortedKeys[rightIndex]][property]) === 'undefined'
        && rightIndex < sortedKeys.length - 1) {
            rightIndex += 1;
        }

        return [sortedKeys[leftIndex], sortedKeys[rightIndex]];
    }

    function interpolate(key, property) {
        const neighbors = findNeighbors(key, property);
        const t = (key - neighbors[0]) / (neighbors[1] - neighbors[0]);
        let a;
        let
            b;

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

    for (i = 0; i < colormap.length; i += 3 + dim) {
        merged[colormap[i]] = {
            r: colormap[i + dim],
            g: colormap[i + dim + 1],
            b: colormap[i + dim + 2],
        };
    }

    if (!opacity) {
        sortedKeys = Object.keys(merged).map(parseFloat).sort((a, b) => a - b);
        opacity = [sortedKeys[0], 1.0, sortedKeys[sortedKeys.length - 1], 1.0];
    }

    for (i = 0; i < opacity.length; i += 2) {
        if (!merged[opacity[i]]) {
            merged[opacity[i]] = {};
        }

        merged[opacity[i]].a = opacity[i + 1];
    }

    sortedKeys = Object.keys(merged).map(parseFloat).sort((a, b) => a - b);

    sortedKeys.forEach((key) => {
        if (typeof (merged[key].a) === 'undefined') {
            merged[key].a = interpolate(key, 'a');
        }

        if (typeof (merged[key].r) === 'undefined') {
            merged[key].r = interpolate(key, 'r');
            merged[key].g = interpolate(key, 'g');
            merged[key].b = interpolate(key, 'b');
        }
    });

    return sortedKeys.reduce((prev, key) => prev.concat(
        [
            parseFloat(key), merged[key].r, merged[key].g, merged[key].b, merged[key].a,
        ],
    ), []);
}

function createSVGGradient(svg, id, colormap, opacity, horizontal) {
    const svgNS = svg.namespaceURI;
    const grad = document.createElementNS(svgNS, 'linearGradient');

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

    const data = mergeColorMapWithOpacity(colormap, opacity, 1);

    const min = data[0];
    const max = data[data.length - 5];

    for (let i = 0; i < data.length; i += 5) {
        const stop = document.createElementNS(svgNS, 'stop');
        const segment = data.slice(i, i + 5);

        stop.setAttribute('offset', ((segment[0] - min) / (max - min)).toString(10));
        stop.setAttribute('stop-color', `rgb(${
            toColor(segment[1])},${
            toColor(segment[2])},${
            toColor(segment[3])})`);
        stop.setAttribute('stop-opacity', segment[4]);
        grad.appendChild(stop);
    }

    const defs = svg.querySelector('defs')
        || svg.insertBefore(document.createElementNS(svgNS, 'defs'), svg.firstChild);

    return defs.appendChild(grad);
}

function createCanvasGradient1d(colorMap, size, opacityFunction) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    let segment;
    let i;

    const merged = mergeColorMapWithOpacity(colorMap, opacityFunction, 1);
    canvas.height = 1;
    canvas.width = size;

    const grd = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    const min = merged[0];
    const max = merged[merged.length - 5];

    for (i = 0; i < merged.length / 5; i++) {
        segment = merged.slice(i * 5, i * 5 + 5);

        grd.addColorStop(
            (segment[0] - min) / (max - min),
            `rgba(${
                toColor(segment[1])}, ${
                toColor(segment[2])}, ${
                toColor(segment[3])}, ${
                segment[4]
            })`,
        );
    }

    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    return canvas;
}

function createCanvasGradient2d(colorMap, size, opacityFunction) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.height = size;
    canvas.width = size;
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);


    function drawRect(v1, v2, v3, v4, sx, sy, w, h) {
        let data = imageData.data;

        sx = Math.round(sx);
        sy = Math.round(sy);
        w = Math.round(w);
        h = Math.round(h);

        for (let y = sy; y < sy + h; y++) {
            let t1 = (y - sy) / h;

            let i1r = v1[2] * (1 - t1) + v4[2] * t1;
            let i1g = v1[3] * (1 - t1) + v4[3] * t1;
            let i1b = v1[4] * (1 - t1) + v4[4] * t1;

            let i2r = v2[2] * (1 - t1) + v3[2] * t1;
            let i2g = v2[3] * (1 - t1) + v3[3] * t1;
            let i2b = v2[4] * (1 - t1) + v3[4] * t1;

            for (let x = sx; x < sx + w; x++) {
                let t2 = (x - sx) / w;

                let r = i1r * (1 - t2) + i2r * t2;
                let g = i1g * (1 - t2) + i2g * t2;
                let b = i1b * (1 - t2) + i2b * t2;

                let ptr = ((size - 1 - y) * size + x) * 4;

                data[ptr] = r * 255;
                data[ptr + 1] = g * 255;
                data[ptr + 2] = b * 255;
                data[ptr + 3] = 255;
            }
        }
    }


    const cm = [];
    let tmp = Array.from(colorMap);
    while (tmp.length) cm.push(tmp.splice(0, 5));

    let v1 = Array.from(new Set(cm.map(function (v) {
        return v[0];
    })));
    let v2 = Array.from(new Set(cm.map(function (v) {
        return v[1];
    })));

    for (let x = 0; x < v1.length - 1; x++) {
        for (let y = 0; y < v2.length - 1; y++) {
            let c1 = cm.filter(function (v) {
                return v[0] == v1[x] && v[1] == v2[y]
            })[0];
            let c2 = cm.filter(function (v) {
                return v[0] == v1[x + 1] && v[1] == v2[y]
            })[0];
            let c3 = cm.filter(function (v) {
                return v[0] == v1[x + 1] && v[1] == v2[y + 1]
            })[0];
            let c4 = cm.filter(function (v) {
                return v[0] == v1[x] && v[1] == v2[y + 1]
            })[0];

            drawRect(c1, c2, c3, c4,
                v1[x] * size, v2[y] * size,
                (v1[x + 1] - v1[x]) * size, (v2[y + 1] - v2[y]) * size);
        }
    }

    ctx.putImageData(imageData, 0, 0);

    // canvas.style = "position: absolute;";
    // document.body.append(canvas);
    return canvas;
}


function createCanvasGradient(colorMap, size, dim, opacityFunction) {
    if (dim === 1) {
        return createCanvasGradient1d(colorMap, size, opacityFunction);
    }

    if (dim === 2) {
        return createCanvasGradient2d(colorMap, size, opacityFunction);
    }

    throw Error("Not supported");
}

function createCanvasColorList(values) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    let i;

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
    createCanvasGradient,
    createCanvasColorList,
    createSVGGradient,
    mergeColorMapWithOpacity,
};
