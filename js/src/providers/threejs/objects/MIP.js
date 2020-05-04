//jshint maxstatements:false,maxcomplexity:false

'use strict';

var THREE = require('three'),
    _ = require('./../../../lodash'),
    colorMapHelper = require('./../../../core/lib/helpers/colorMap'),
    typedArrayToThree = require('./../helpers/Fn').typedArrayToThree,
    areAllChangesResolve = require('./../helpers/Fn').areAllChangesResolve,
    commonUpdate = require('./../helpers/Fn').commonUpdate;

/**
 * Loader strategy to handle Volume object
 * @method Volume
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @param {K3D}
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config) {
        config.samples = config.samples || 512.0;
        config.gradient_step = config.gradient_step || 0.005;

        var geometry = new THREE.BoxBufferGeometry(1, 1, 1),
            modelMatrix = new THREE.Matrix4(),
            translation = new THREE.Vector3(),
            rotation = new THREE.Quaternion(),
            scale = new THREE.Vector3(),
            colorMap = (config.color_map && config.color_map.data) || null,
            opacityFunction = (config.opacity_function && config.opacity_function.data) || null,
            colorRange = config.color_range,
            samples = config.samples,
            object,
            texture,
            jitterTexture;

        if (opacityFunction === null) {
            opacityFunction = [colorMap[0], 0.0, colorMap[colorMap.length - 4], 1.0];
        }

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        modelMatrix.decompose(translation, rotation, scale);

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
        texture.wrapS = texture.wrapT = THREE.MirroredRepeatWrapping;
        texture.needsUpdate = true;

        jitterTexture = new THREE.DataTexture(
            new Uint8Array(_.range(64 * 64).map(function () {
                return 255.0 * Math.random();
            })),
            64, 64, THREE.RedFormat, THREE.UnsignedByteType);
        jitterTexture.minFilter = THREE.LinearFilter;
        jitterTexture.magFilter = THREE.LinearFilter;
        jitterTexture.wrapS = jitterTexture.wrapT = THREE.ClampToEdgeWrapping;
        jitterTexture.generateMipmaps = false;
        jitterTexture.needsUpdate = true;

        var canvas = colorMapHelper.createCanvasGradient(colorMap, 1024, opacityFunction);
        var colormap = new THREE.CanvasTexture(canvas, THREE.UVMapping, THREE.ClampToEdgeWrapping,
            THREE.ClampToEdgeWrapping, THREE.NearestFilter, THREE.NearestFilter);
        colormap.needsUpdate = true;

        var uniforms = {
            volumeMapSize: {value: new THREE.Vector3(config.volume.shape[2], config.volume.shape[1], config.volume.shape[0])},
            low: {value: colorRange[0]},
            high: {value: colorRange[1]},
            gradient_step: {value: config.gradient_step},
            samples: {value: samples},
            translation: {value: translation},
            rotation: {value: rotation},
            scale: {value: scale},
            volumeTexture: {type: 't', value: texture},
            colormap: {type: 't', value: colormap},
            jitterTexture: {type: 't', value: jitterTexture}
        };

        var material = new THREE.ShaderMaterial({
            uniforms: _.merge(
                uniforms,
                THREE.UniformsLib.lights
            ),
            defines: {
                USE_SPECULAR: 1
            },
            vertexShader: require('./shaders/MIP.vertex.glsl'),
            fragmentShader: require('./shaders/MIP.fragment.glsl'),
            side: THREE.BackSide,
            depthTest: false,
            depthWrite: false,
            lights: true,
            clipping: true,
            transparent: true
        });

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        object = new THREE.Mesh(geometry, material);
        object.applyMatrix(modelMatrix);
        object.updateMatrixWorld();

        object.onRemove = function () {
            object.material.uniforms.volumeTexture.value = undefined;
            object.material.uniforms.colormap.value.dispose();
            object.material.uniforms.colormap.value = undefined;
            jitterTexture.dispose();
            jitterTexture = undefined;
        };

        return Promise.resolve(object);
    },

    update: function (config, changes, obj) {
        var resolvedChanges = {};

        if (typeof(changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
            obj.material.uniforms.low.value = changes.color_range[0];
            obj.material.uniforms.high.value = changes.color_range[1];

            resolvedChanges.color_range = null;
        }


        if (typeof(changes.volume) !== 'undefined' && !changes.volume.timeSeries) {
            obj.material.uniforms.volumeTexture.value.image.data = changes.volume.data;
            obj.material.uniforms.volumeTexture.value.needsUpdate = true;

            resolvedChanges.volume = null;
        }

        if ((typeof(changes.color_map) !== 'undefined' && !changes.color_map.timeSeries) ||
            (typeof(changes.opacity_function) !== 'undefined' && !changes.opacity_function.timeSeries)) {

            var canvas = colorMapHelper.createCanvasGradient(
                (changes.color_map && changes.color_map.data) || config.color_map.data,
                1024,
                (changes.opacity_function && changes.opacity_function.data) || config.opacity_function.data
            );

            obj.material.uniforms.colormap.value.image = canvas;
            obj.material.uniforms.colormap.value.needsUpdate = true;

            resolvedChanges.color_map = null;
            resolvedChanges.opacity_function = null;
        }

        ['samples', 'gradient_step'].forEach(function (key) {
            if (changes[key] && !changes[key].timeSeries) {
                obj.material.uniforms[key].value = changes[key];
                resolvedChanges[key] = null;
            }
        });

        commonUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
};
