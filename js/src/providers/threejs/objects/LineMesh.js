'use strict';

var THREE = require('three'),
    Fn = require('./../helpers/Fn'),
    areAllChangesResolve = Fn.areAllChangesResolve,
    modelMatrixUpdate = Fn.modelMatrixUpdate,
    colorsToFloat32Array = require('./../../../core/lib/helpers/buffer').colorsToFloat32Array,
    streamLine = require('./../helpers/Streamline'),
    handleColorMap = Fn.handleColorMap;

/**
 * Loader strategy to handle Line object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config) {
        config.radial_segments = typeof (config.radial_segments) !== 'undefined' ? config.radial_segments : 8;
        config.width = typeof (config.width) !== 'undefined' ? config.width : 0.1;

        var geometry,
            material = new THREE.MeshPhongMaterial({
                emissive: 0,
                shininess: 50,
                specular: 0x111111,
                side: THREE.DoubleSide,
                wireframe: false
            }),
            radialSegments = config.radial_segments,
            width = config.width,
            verticesColors = (config.colors && config.colors.data) || null,
            color = new THREE.Color(config.color),
            colorRange = config.color_range,
            colorMap = (config.color_map && config.color_map.data) || null,
            attribute = (config.attribute && config.attribute.data) || null,
            object,
            modelMatrix = new THREE.Matrix4(),
            position = config.vertices.data;

        if (verticesColors && verticesColors.length === position.length / 3) {
            verticesColors = colorsToFloat32Array(verticesColors);
        }

        geometry = streamLine(position, attribute, width, radialSegments, color, verticesColors, colorRange);

        if (attribute && colorRange && colorMap && attribute.length > 0 && colorRange.length > 0 && colorMap.length > 0) {
            handleColorMap(geometry, colorMap, colorRange, null, material);
        } else {
            material.setValues({vertexColors: THREE.VertexColors});
        }

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);

        object = new THREE.Mesh(geometry, material);
        object.applyMatrix(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },

    update: function (config, changes, obj) {
        modelMatrixUpdate(config, changes, obj);

        if (areAllChangesResolve(changes)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
};
