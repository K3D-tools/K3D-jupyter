'use strict';

var THREE = require('three'),
    intersectHelper = require('./../helpers/Intersection'),
    colorMapHelper = require('./../../../core/lib/helpers/colorMap'),
    typedArrayToThree = require('./../helpers/Fn').typedArrayToThree,
    areAllChangesResolve = require('./../helpers/Fn').areAllChangesResolve,
    commonUpdate = require('./../helpers/Fn').commonUpdate,
    getSide = require('./../helpers/Fn').getSide;

/**
 * Loader strategy to handle Mesh object
 * @method Mesh
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config, K3D) {
        config.color = typeof (config.color) !== 'undefined' ? config.color : 255;
        config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;

        var modelMatrix = new THREE.Matrix4(),
            colorRange = config.color_range,
            colorMap = (config.color_map && config.color_map.data) || null,
            vertices = (config.vertices && config.vertices.data) || null,
            indices = (config.indices && config.indices.data) || null,
            opacityFunction = null,
            geometry = new THREE.BufferGeometry(),
            texture,
            object,
            material;

        if (config.opacity_function && config.opacity_function.data && config.opacity_function.data.length > 0) {
            opacityFunction = config.opacity_function.data;
        }

        var canvas = colorMapHelper.createCanvasGradient(colorMap, 1024, opacityFunction);
        var colormap = new THREE.CanvasTexture(canvas, THREE.UVMapping, THREE.ClampToEdgeWrapping,
            THREE.ClampToEdgeWrapping, THREE.NearestFilter, THREE.NearestFilter);
        colormap.needsUpdate = true;

        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));

        texture = new THREE.DataTexture3D(
            config.volume.data,
            config.volume.shape[2],
            config.volume.shape[1],
            config.volume.shape[0]);
        texture.format = THREE.RedFormat;
        texture.type = typedArrayToThree(config.volume.data.constructor);

        texture.generateMipmaps = false;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.needsUpdate = true;

        material = new THREE.ShaderMaterial({
            uniforms: {
                opacity: {value: config.opacity},
                low: {value: colorRange[0]},
                high: {value: colorRange[1]},
                volumeTexture: {type: 't', value: texture},
                colormap: {type: 't', value: colormap},
                b1: {
                    type: 'v3',
                    value: new THREE.Vector3(
                        config.volume_bounds.data[0],
                        config.volume_bounds.data[2],
                        config.volume_bounds.data[4])
                },
                b2: {
                    type: 'v3',
                    value: new THREE.Vector3(
                        config.volume_bounds.data[1],
                        config.volume_bounds.data[3],
                        config.volume_bounds.data[5])
                }
            },
            side: getSide(config),
            vertexShader: require('./shaders/MeshVolume.vertex.glsl'),
            fragmentShader: require('./shaders/MeshVolume.fragment.glsl'),
            depthTest: (config.opacity === 1.0 && opacityFunction === null),
            transparent: (config.opacity !== 1.0 || opacityFunction !== null),
            lights: false,
            clipping: true
        });

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

        intersectHelper.update(config, changes, resolvedChanges, obj, K3D);

        commonUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
};
