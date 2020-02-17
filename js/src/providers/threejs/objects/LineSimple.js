'use strict';

var THREE = require('three'),
    colorsToFloat32Array = require('./../../../core/lib/helpers/buffer').colorsToFloat32Array,
    Fn = require('./../helpers/Fn'),
    commonUpdate = Fn.commonUpdate,
    areAllChangesResolve = Fn.areAllChangesResolve,
    getColorsArray = Fn.getColorsArray,
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

        var geometry = new THREE.BufferGeometry(),
            material = new THREE.MeshBasicMaterial(),
            verticesColors = (config.colors && config.colors.data) || null,
            color = new THREE.Color(config.color),
            colors,
            colorRange = config.color_range,
            colorMap = (config.color_map && config.color_map.data) || null,
            attribute = (config.attribute && config.attribute.data) || null,
            object = new THREE.Line(geometry, material),
            modelMatrix = new THREE.Matrix4(),
            position = config.vertices.data;

        if (attribute && colorRange && colorMap && attribute.length > 0 && colorRange.length > 0 && colorMap.length > 0) {
            handleColorMap(geometry, colorMap, colorRange, attribute, material);
        } else {
            colors = (verticesColors && verticesColors.length === position.length / 3 ?
                    colorsToFloat32Array(verticesColors) : getColorsArray(color, position.length / 3)
            );

            material.setValues({vertexColors: THREE.VertexColors});
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },

    update: function (config, changes, obj) {
        var resolvedChanges = {};

        if (typeof (obj.geometry.attributes.uv) !== 'undefined') {
            if (typeof(changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
                obj.material.uniforms.low.value = changes.color_range[0];
                obj.material.uniforms.high.value = changes.color_range[1];

                resolvedChanges.color_range = null;
            }

            if (typeof(changes.attribute) !== 'undefined' && !changes.attribute.timeSeries &&
                changes.attribute.data.length === obj.geometry.attributes.uv.array.length) {

                var data = obj.geometry.attributes.uv.array;

                for (var i = 0; i < data.length; i++) {
                    data[i] = (changes.attribute.data[i] - config.color_range[0]) /
                              (config.color_range[1] - config.color_range[0]);
                }

                obj.geometry.attributes.uv.needsUpdate = true;
                resolvedChanges.attribute = null;
            }
        }

        if (typeof(changes.vertices) !== 'undefined' && !changes.vertices.timeSeries &&
            changes.vertices.data.length === obj.geometry.attributes.position.array.length) {
            obj.geometry.attributes.position.array.set(changes.vertices.data);
            obj.geometry.attributes.position.needsUpdate = true;
            resolvedChanges.vertices = null;
        }

        commonUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
};
