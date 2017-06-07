'use strict';

var buffer = require('./../../../core/lib/helpers/buffer'),
    getTwoColorsArray = require('./../helpers/Fn').getTwoColorsArray,
    generateArrow = require('./../helpers/Fn').generateArrow,
    Text2d = require('./Text2d');

/**
 * Loader strategy to handle Vectors  object
 * @method Vector
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @param {K3D}
 */
module.exports = function (config, K3D) {
    var originColor = new THREE.Color(config.origin_color || 255),
        headColor = new THREE.Color(config.head_color || 255),
        vectors = config.vectors,
        origins = config.origins,
        colors = config.colors || [],
        useHead = config.use_head || true,
        headSize = config.head_size || 1.0,
        object = new THREE.Group(),
        origin,
        destination,
        i,
        labelSize = config.label_size,
        labels = config.labels,
        labelsObjects = [],
        heads = null,
        singleConeGeometry,
        linesGeometry = new THREE.BufferGeometry(),
        lineVertices = [];

    colors = colors.length > 0 ? buffer.colorsToFloat32Array(colors) :
        getTwoColorsArray(originColor, headColor, vectors.length / 3 * 2);

    if (vectors.length !== origins.length) {
        throw new Error('vectors and origins should have the same length');
    }

    if (colors && colors.length / 2 !== vectors.length) {
        throw new Error('there should be 2 colors for each vector');
    }

    singleConeGeometry = new THREE.CylinderGeometry(0, 0.025 * headSize, 0.2 * headSize, 5, 1)
        .translate(0, -0.1 * headSize, 0);

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
                    createText(labels[i / 3], origin, destination, labelSize, K3D)
                );
            }
        }
    }

    if (useHead) {
        addHeads(heads, object);
    }

    linesGeometry.addAttribute('position', new THREE.BufferAttribute(new Float32Array(lineVertices), 3));
    linesGeometry.addAttribute('color', new THREE.BufferAttribute(colors, 3));

    linesGeometry.computeBoundingSphere();
    linesGeometry.computeBoundingBox();

    object.add(new THREE.LineSegments(linesGeometry, new THREE.LineBasicMaterial({
        vertexColors: THREE.VertexColors,
        linewidth: config.line_width || 1.0
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
    headsGeometry.computeBoundingBox();

    object.add(
        new THREE.Mesh(
            headsGeometry,
            new THREE.MeshBasicMaterial({vertexColors: THREE.VertexColors})
        )
    );
}

function createText(text, origin, destination, labelSize, K3D) {
    var center = origin.clone().add(destination).divideScalar(2),
        textConfig = {
            'position': [center.x, center.y, center.z],
            'referencePoint': 'cb',
            'text': text,
            'size': labelSize
        };

    return new Text2d(textConfig, K3D);
}
