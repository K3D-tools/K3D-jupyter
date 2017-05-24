'use strict';

var buffer = require('./../../../core/lib/helpers/buffer');
var lut = require('./../../../core/lib/helpers/lut');

/**
 * Loader strategy to handle Mesh object
 * @method STL
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {K3D.Config} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var modelViewMatrix = new THREE.Matrix4(),
        material = new THREE.MeshPhongMaterial({
            color: config.get('color'),
            emissive: 0,
            shininess: 50,
            specular: 0x111111,
            side: THREE.DoubleSide,
            shading: THREE.FlatShading
        }),
        colorRange = config.get('color_range'),
        colorMap = config.get('color_map'),
        vertexScalars = config.get('vertex_scalars'),
        vertices = config.get('vertices'),
        indices = config.get('indices'),
        geometry = new THREE.BufferGeometry(),
        object;

    if (typeof (colorRange) === 'string') {
        colorRange = buffer.base64ToArrayBuffer(colorRange);
    }

    if (typeof (colorMap) === 'string') {
        colorMap = buffer.base64ToArrayBuffer(colorMap);
    }

    if (typeof (vertexScalars) === 'string') {
        vertexScalars = buffer.base64ToArrayBuffer(vertexScalars);
    }

    if (typeof (vertices) === 'string') {
        vertices = buffer.base64ToArrayBuffer(vertices);
    }

    if (typeof (indices) === 'string') {
        indices = buffer.base64ToArrayBuffer(indices);
    }

    if (typeof(colorRange) !== 'undefined' && typeof(colorMap) !== 'undefined' &&
        typeof(vertexScalars) !== 'undefined') {

        // ColorMap variant
        colorRange = new Float32Array(colorRange);
        colorMap = new Float32Array(colorMap);

        var canvas = lut(colorMap, 1024);

        var texture = new THREE.CanvasTexture(canvas, THREE.UVMapping, THREE.ClampToEdgeWrapping,
            THREE.ClampToEdgeWrapping, THREE.NearestFilter, THREE.NearestFilter);
        texture.needsUpdate = true;

        material = new THREE.MeshPhongMaterial({
            color: 0xffffff,
            emissive: 0,
            shininess: 50,
            specular: 0x111111,
            map: texture,
            side: THREE.DoubleSide,
            shading: THREE.FlatShading
        });

        vertexScalars = new Float32Array(vertexScalars);

        var uvs = new Float32Array(vertexScalars.length);

        for (var i = 0; i < vertexScalars.length; i++) {
            if (vertexScalars[i] < colorRange[0]) {
                uvs[i] = 0.0;
            } else if (vertexScalars[i] > colorRange[1]) {
                uvs[i] = 1.0;
            } else {
                uvs[i] = (vertexScalars[i] - colorRange[0]) / (colorRange[1] - colorRange[0]);
            }
        }

        geometry.addAttribute('uv', new THREE.BufferAttribute(uvs, 1));
    }

    geometry.addAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    object = new THREE.Mesh(geometry, material);

    modelViewMatrix.set.apply(modelViewMatrix, config.get('modelViewMatrix'));
    object.applyMatrix(modelViewMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
