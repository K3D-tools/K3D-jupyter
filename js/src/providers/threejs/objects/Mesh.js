'use strict';

var handleColorMap = require('./../helpers/Fn').handleColorMap;

/**
 * Loader strategy to handle Mesh object
 * @method STL
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 255;
        config.wireframe = typeof (config.wireframe) !== 'undefined' ? config.wireframe : false;
        config.flat_shading = typeof (config.flat_shading) !== 'undefined' ? config.flat_shading : true;

        var modelMatrix = new THREE.Matrix4(),
            MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial,
            material = new MaterialConstructor({
                color: config.color,
                emissive: 0,
                shininess: 50,
                specular: 0x111111,
                side: THREE.DoubleSide,
                flatShading: config.flat_shading,
                wireframe: config.wireframe
            }),
            colorRange = config.color_range,
            colorMap = (config.color_map && config.color_map.data) || null,
            attribute = (config.attribute && config.attribute.data) || null,
            vertices = (config.vertices && config.vertices.data) || null,
            indices = (config.indices && config.indices.data) || null,
            geometry = new THREE.BufferGeometry(),
            object;

        if (attribute && colorRange && colorMap && attribute.length > 0 && colorRange.length > 0 && colorMap.length > 0) {
            handleColorMap(geometry, colorMap, colorRange, attribute, material);
        }

        geometry.addAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));

        if (config.flat_shading === false) {
            geometry.computeVertexNormals();
        }

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        object = new THREE.Mesh(geometry, material);

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    }

    // update: function (config, prevConfig, obj, K3D) {
    //     console.log(config, prevConfig, obj, K3D);
    //
    //     return false;
    // }
};
