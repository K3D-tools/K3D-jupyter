'use strict';

var lut = require('./../../../core/lib/helpers/lut');

function handleColorMap(geometry, colorMap, colorRange, attributes, material) {
    var canvas, texture, uvs, i;

    canvas = lut(colorMap, 1024);

    texture = new THREE.CanvasTexture(canvas, THREE.UVMapping, THREE.ClampToEdgeWrapping,
        THREE.ClampToEdgeWrapping, THREE.NearestFilter, THREE.NearestFilter);
    texture.needsUpdate = true;

    material.setValues({
        map: texture,
        color: 0xffffff
    });

    uvs = new Float32Array(attributes.length);

    for (i = 0; i < attributes.length; i++) {
        uvs[i] = (attributes[i] - colorRange[0]) / (colorRange[1] - colorRange[0]);
    }

    geometry.addAttribute('uv', new THREE.BufferAttribute(uvs, 1));
}

/**
 * Loader strategy to handle Mesh object
 * @method STL
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var modelMatrix = new THREE.Matrix4(),
        MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial,
        material = new MaterialConstructor({
            color: config.color,
            emissive: 0,
            shininess: 25,
            specular: 0x111111,
            side: THREE.DoubleSide,
            shading: THREE.FlatShading,
            wireframe: config.wireframe || false
        }),
        colorRange = config.color_range,
        colorMap = (config.color_map && config.color_map.buffer) || null,
        attribute = (config.attribute && config.attribute.buffer) || null,
        vertices = (config.vertices && config.vertices.buffer) || null,
        indices = (config.indices && config.indices.buffer) || null,
        geometry = new THREE.BufferGeometry(),
        object;

    if (attribute && colorRange && colorMap && attribute.length > 0 && colorRange.length > 0 && colorMap.length > 0) {
        handleColorMap(geometry, colorMap, colorRange, attribute, material);
    }

    geometry.addAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    object = new THREE.Mesh(geometry, material);

    modelMatrix.set.apply(modelMatrix, config.model_matrix.buffer);
    object.applyMatrix(modelMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
