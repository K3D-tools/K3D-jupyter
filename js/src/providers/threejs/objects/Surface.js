'use strict';

var THREE = require('three'),
    intersectHelper = require('./../helpers/Intersection'),
    handleColorMap = require('./../helpers/Fn').handleColorMap,
    areAllChangesResolve = require('./../helpers/Fn').areAllChangesResolve,
    modelMatrixUpdate = require('./../helpers/Fn').modelMatrixUpdate;

/**
 * Loader strategy to handle Surface object
 * @method MarchingCubes
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

        var heights = config.heights.data,
            width = config.heights.shape[1],
            height = config.heights.shape[0],
            modelMatrix = new THREE.Matrix4(),
            MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial,
            colorRange = config.color_range,
            colorMap = (config.color_map && config.color_map.data) || null,
            attribute = (config.attribute && config.attribute.data) || null,
            material = new MaterialConstructor({
                color: config.color,
                emissive: 0,
                shininess: 50,
                specular: 0x111111,
                side: THREE.DoubleSide,
                flatShading: config.flat_shading,
                wireframe: config.wireframe
            }),
            geometry = new THREE.BufferGeometry(),
            vertices = new Float32Array(width * height * 3),
            indices = [],
            object,
            x, y, i, p;

        for (y = 0, i = 0, p = 0; y < height; y++) {
            for (x = 0; x < width; x++, p++, i += 3) {
                vertices[i] = x / (width - 1);
                vertices[i + 1] = y / (height - 1);
                vertices[i + 2] = heights[p];
            }
        }

        for (y = 0, i = 0; y < height - 1; y++) {
            for (x = 0; x < width - 1; x++, i += 6) {
                indices[i] = x + y * width;
                indices[i + 1] = indices[i + 3] = x + 1 + y * width;
                indices[i + 2] = indices[i + 5] = x + (y + 1) * width;
                indices[i + 4] = x + 1 + (y + 1) * width;
            }
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(indices);

        if (config.flat_shading === false) {
            geometry.computeVertexNormals();
        }

        if (attribute && colorRange && colorMap && attribute.length > 0 && colorRange.length > 0 && colorMap.length > 0) {
            handleColorMap(geometry, colorMap, colorRange, attribute, material);
        }

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        object = new THREE.Mesh(geometry, material);

        intersectHelper.init(config, object, K3D);

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);

        object.position.set(-0.5, -0.5, 0);
        object.updateMatrix();

        object.applyMatrix(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },

    update: function (config, changes, obj, K3D) {
        var resolvedChanges = {};

        intersectHelper.update(config, changes, resolvedChanges, obj, K3D);
        modelMatrixUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
};
