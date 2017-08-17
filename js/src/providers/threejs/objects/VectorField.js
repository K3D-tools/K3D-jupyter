'use strict';

var buffer = require('./../../../core/lib/helpers/buffer'),
    getTwoColorsArray = require('./../helpers/Fn').getTwoColorsArray,
    generateArrow = require('./../helpers/Fn').generateArrow;

/**
 * Loader strategy to handle Vector Fields object
 * @method Vector
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 */
module.exports = function (config) {
    var modelMatrix = new THREE.Matrix4().fromArray(config.model_matrix.buffer),
        originColor = new THREE.Color(config.origin_color || 255),
        headColor = new THREE.Color(config.head_color || 255),
        width = config.vectors.shape[2],
        height = config.vectors.shape[1],
        length = config.vectors.shape[0],
        vectors = config.vectors.buffer,
        colors = (config.colors && config.colors.buffer) || null,
        useHead = !(config.use_head === false),
        headSize = config.head_size || 1.0,
        scale = config.scale || 1.0,
        object = new THREE.Group(),
        x,
        y,
        z,
        i,
        scalar,
        origin,
        destination,
        heads = null,
        singleConeGeometry,
        linesGeometry = new THREE.BufferGeometry(),
        lineVertices = [],
        colorsToFloat32Array = buffer.colorsToFloat32Array;

    if (config.vectors.shape.length === 3)
    {
        // 2d vectors fields
        width = height;
        height = length;
        length = 1;
        vectors = convert2DVectorsTable(vectors);
    }

    scalar = scale / Math.max(width, height, length);
    colors = colors ? colorsToFloat32Array(colors) :
        getTwoColorsArray(originColor, headColor, width * height * length * 2);
    singleConeGeometry = new THREE.CylinderGeometry(0, 0.025 * headSize * scalar, 0.2 * headSize * scalar, 5, 1)
        .translate(0, -0.1 * headSize * scalar, 0);

    for (z = 0, i = 0; z < length; z++) {
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++, i++) {

                origin = new THREE.Vector3(x / width, y / height, z / length);
                destination = new THREE.Vector3(
                    (vectors[i * 3] / 2) * scalar,
                    (vectors[i * 3 + 1] / 2) * scalar,
                    (vectors[i * 3 + 2] / 2) * scalar
                ).add(origin);

                heads = generateArrow(
                    useHead ? new THREE.Geometry().copy(singleConeGeometry) : null,
                    lineVertices,
                    heads,
                    origin,
                    destination,
                    new THREE.Color(colors[i * 6 + 3], colors[i * 6 + 4], colors[i * 6 + 5])
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
        linewidth: config.lineWidth || 1
    })));

    object.position.set(-0.5, -0.5, length === 1 ? 0 : -0.5);
    object.updateMatrix();
    object.applyMatrix(modelMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
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

function convert2DVectorsTable(vectors) {
    var i, tempVectors = new Float32Array(vectors.length / 2 * 3);

    for (i = 0; i < vectors.length / 2; i++) {
        tempVectors[i * 3] = vectors[i * 2];
        tempVectors[i * 3 + 1] = vectors[i * 2 + 1];
        tempVectors[i * 3 + 2] = 0.0;
    }

    return tempVectors;
}
