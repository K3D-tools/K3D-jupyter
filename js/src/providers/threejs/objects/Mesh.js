'use strict';

var THREE = require('three'),
    intersectHelper = require('./../helpers/Intersection'),
    handleColorMap = require('./../helpers/Fn').handleColorMap,
    areAllChangesResolve = require('./../helpers/Fn').areAllChangesResolve,
    modelMatrixUpdate = require('./../helpers/Fn').modelMatrixUpdate;

/**
 * Loader strategy to handle Mesh object
 * @method Mesh
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config, K3D) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 255;
        config.wireframe = typeof (config.wireframe) !== 'undefined' ? config.wireframe : false;
        config.flat_shading = typeof (config.flat_shading) !== 'undefined' ? config.flat_shading : true;
        config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;

        var modelMatrix = new THREE.Matrix4(),
            MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial,
            material = new MaterialConstructor({
                color: config.color,
                emissive: 0,
                shininess: 50,
                specular: 0x111111,
                side: THREE.DoubleSide,
                flatShading: config.flat_shading,
                wireframe: config.wireframe,
                opacity: config.opacity,
                depthTest: config.opacity === 1.0,
                depthWrite: config.opacity === 1.0,
                transparent: config.opacity !== 1.0
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

        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));

        if (config.flat_shading === false) {
            geometry.computeVertexNormals();
        }

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        object = new THREE.Mesh(geometry, material);

        intersectHelper.init(config, object, K3D);

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },

    update: function (config, changes, obj, K3D) {
        var resolvedChanges = {};

        if (typeof (obj.geometry.attributes.uv) !== 'undefined') {
            if (typeof(changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
                obj.material.uniforms.low.value = changes.color_range[0];
                obj.material.uniforms.high.value = changes.color_range[1];

                resolvedChanges.color_range = null;
            }

            if (typeof(changes.attribute) !== 'undefined' && !changes.attribute.timeSeries) {
                var data = obj.geometry.attributes.uv.array;

                for (var i = 0; i < data.length; i++) {
                    data[i] = (changes.attribute.data[i] - config.color_range[0]) /
                              (config.color_range[1] - config.color_range[0]);
                }

                obj.geometry.attributes.uv.needsUpdate = true;
                resolvedChanges.attribute = null;
            }
        }

        intersectHelper.update(config, changes, resolvedChanges, obj, K3D);
        modelMatrixUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
};
