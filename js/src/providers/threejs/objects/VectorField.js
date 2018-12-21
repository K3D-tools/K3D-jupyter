'use strict';

var buffer = require('./../../../core/lib/helpers/buffer'),
    MeshLine = require('./../helpers/THREE.MeshLine'),
    getTwoColorsArray = require('./../helpers/Fn').getTwoColorsArray,
    generateArrow = require('./../helpers/Fn').generateArrow;

/**
 * Loader strategy to handle Vector Fields object
 * @method Vector
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 */
module.exports = {
    create: function (config, K3D) {
        config.visible = typeof(config.visible) !== 'undefined' ? config.visible : true;
        config.origin_color = typeof(config.origin_color) !== 'undefined' ? config.origin_color : 255;
        config.head_color = typeof(config.head_color) !== 'undefined' ? config.head_color : 255;
        config.use_head = typeof(config.use_head) !== 'undefined' ? config.use_head : true;
        config.head_size = config.head_size || 1.0;
        config.scale = config.scale || 1.0;
        config.line_width = config.line_width || 0.01;

        var modelMatrix = new THREE.Matrix4().fromArray(config.model_matrix.data),
            originColor = new THREE.Color(config.origin_color),
            headColor = new THREE.Color(config.head_color),
            width = config.vectors.shape[2],
            height = config.vectors.shape[1],
            length = config.vectors.shape[0],
            vectors = config.vectors.data,
            colors = (config.colors && config.colors.data) || null,
            useHead = config.use_head,
            headSize = config.head_size,
            scale = config.scale,
            object = new THREE.Group(),
            x, y, z, i,
            scalar, origin, destination,
            heads = null,
            resizelistenerId,
            singleConeGeometry,
            lineVertices = [],
            colorsToFloat32Array = buffer.colorsToFloat32Array;

        if (config.vectors.shape.length === 3) {
            // 2d vectors fields
            width = height;
            height = length;
            length = 1;
            vectors = convert2DVectorsTable(vectors);
        }

        scalar = scale / Math.max(width, height, length);
        colors = colors ? colorsToFloat32Array(colors) :
            getTwoColorsArray(originColor, headColor, width * height * length * 2);
        singleConeGeometry = new THREE.CylinderBufferGeometry(0, 0.025 * headSize * scalar, 0.2 * headSize * scalar, 5, 1)
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
                        useHead ? new THREE.Geometry().fromBufferGeometry(singleConeGeometry) : null,
                        lineVertices,
                        heads,
                        origin,
                        destination,
                        new THREE.Color(colors[i * 6 + 3], colors[i * 6 + 4], colors[i * 6 + 5]),
                        0.2 * headSize * scalar
                    );
                }
            }
        }

        if (useHead) {
            addHeads(heads, object);
        }

        var line = new MeshLine.MeshLine();
        var material = new MeshLine.MeshLineMaterial({
            color: new THREE.Color(1, 1, 1),
            opacity: 1.0,
            sizeAttenuation: true,
            transparent: true,
            lineWidth: config.line_width,
            resolution: new THREE.Vector2(K3D.getWorld().width, K3D.getWorld().height),
            side: THREE.DoubleSide
        });

        line.setGeometry(new Float32Array(lineVertices), true, null, colors);
        line.geometry.computeBoundingSphere();
        line.geometry.computeBoundingBox();

        line = new THREE.Mesh(line.geometry, material);
        object.add(line);

        object.position.set(-0.5, -0.5, length === 1 ? 0 : -0.5);
        object.updateMatrix();
        object.applyMatrix(modelMatrix);
        object.boundingBox = line.geometry.boundingBox;
        object.updateMatrixWorld();

        resizelistenerId = K3D.on(K3D.events.RESIZED, function () {
            line.material.uniforms.resolution.value.x = K3D.getWorld().width;
            line.material.uniforms.resolution.value.y = K3D.getWorld().height;
        });

        object.onRemove = function () {
            K3D.off(K3D.events.RESIZED, resizelistenerId);
        };

        return Promise.resolve(object);
    }
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
