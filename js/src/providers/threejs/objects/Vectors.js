'use strict';

var buffer = require('./../../../core/lib/helpers/buffer'),
    getTwoColorsArray = require('./../helpers/Fn').getTwoColorsArray,
    generateArrow = require('./../helpers/Fn').generateArrow,
    Text2d = require('./Text2d'),
    Config = require('./../../../core/lib/Config');

/**
 * Loader strategy to handle Vectors  object
 * @method Vector
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {K3D.Config} config all configurations params from JSON
 * @param {K3D}
 */
module.exports = function (config, K3D) {
    var originColor = new THREE.Color(config.get('originColor', 255)),
        headColor = new THREE.Color(config.get('headColor', 255)),
        vectors = config.get('vectors'),
        origins = config.get('origins'),
        colors = config.get('colors'),
        useHead = config.get('useHead', true),
        object = new THREE.Group(),
        origin,
        destination,
        i,
        labels = config.get('labels'),
        labelsObjects = [],
        labelsSize = config.get('labelsSize'),
        heads = null,
        singleConeGeometry,
        linesGeometry = new THREE.BufferGeometry(),
        lineVertices = [],
        toFloat32Array = buffer.toFloat32Array,
        colorsToFloat32Array = buffer.colorsToFloat32Array;

    if (typeof (vectors) === 'string') {
        vectors = buffer.base64ToArrayBuffer(vectors);
    }

    if (typeof (origins) === 'string') {
        origins = buffer.base64ToArrayBuffer(origins);
    }

    if (typeof (colors) === 'string') {
        colors = buffer.base64ToArrayBuffer(colors);
    }

    vectors = toFloat32Array(vectors);
    origins = toFloat32Array(origins);
    colors = colors ? colorsToFloat32Array(colors) :
        getTwoColorsArray(originColor, headColor, vectors.length / 3 * 2);

    if (vectors.length !== origins.length) {
        throw new Error('vectors and origins should have the same length');
    }

    if (colors && colors.length / 2 !== vectors.length) {
        throw new Error('there should be 2 colors for each vector');
    }

    singleConeGeometry = new THREE.CylinderGeometry(0, 0.025, 0.2, 5, 1).translate(0, -0.1, 0);

    for (i = 0; i < vectors.length; i += 3) {
        origin = new THREE.Vector3(origins[i], origins[i + 1], origins[i + 2]);
        destination = new THREE.Vector3(vectors[i], vectors[i + 1], vectors[i + 2]).add(origin);

        heads = generateArrow(
            useHead ? new THREE.Geometry().copy(singleConeGeometry) : null,
            lineVertices,
            heads,
            origin,
            destination,
            new THREE.Color(colors[i * 2 + 3], colors[i * 2 + 4], colors[i * 2 + 5])
        );

        if (labels) {
            if (labels[i / 3]) {
                labelsObjects.push(
                    createText(labels[i / 3], origin, destination, labelsSize, K3D)
                );
            }
        }
    }

    if (useHead) {
        addHeads(heads, object);
    }

    linesGeometry.addAttribute('position', new THREE.BufferAttribute(toFloat32Array(lineVertices), 3));
    linesGeometry.addAttribute('color', new THREE.BufferAttribute(colors, 3));

    linesGeometry.computeBoundingSphere();

    object.add(new THREE.LineSegments(linesGeometry, new THREE.LineBasicMaterial({
        vertexColors: THREE.VertexColors,
        linewidth: config.get('lineWidth', 1)
    })));

    object.updateMatrixWorld();

    return Promise.all(labelsObjects).then(function (texts) {
        texts.forEach(function (text) {
            object.add(text);
        });

        object.onRemove = function () {
            texts.forEach(function (text) {
                text.onRemove();
            });
        };

        return object;
    });
};

function addHeads(heads, object) {
    var headsGeometry = new THREE.BufferGeometry().fromGeometry(heads);
    headsGeometry.computeBoundingSphere();

    object.add(
        new THREE.Mesh(
            headsGeometry,
            new THREE.MeshBasicMaterial({vertexColors: THREE.VertexColors})
        )
    );
}

function createText(text, origin, destination, labelsSize, K3D) {
    var center = origin.clone().add(destination).divideScalar(2),
        textConfig = {
            'position': [center.x, center.y, center.z],
            'referencePoint': 'cb',
            'text': text,
            'size': labelsSize
        };

    return new Text2d(new Config(textConfig), K3D);
}
