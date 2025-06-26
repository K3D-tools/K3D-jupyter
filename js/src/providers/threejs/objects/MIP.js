// jshint maxstatements:false,maxcomplexity:false

const THREE = require('three');
const _ = require('../../../lodash');
const colorMapHelper = require('../../../core/lib/helpers/colorMap');
const {typedArrayToThree} = require('../helpers/Fn');
const {areAllChangesResolve} = require('../helpers/Fn');
const {commonUpdate} = require('../helpers/Fn');
const {ensure256size} = require('../helpers/Fn');

/**
 * Loader strategy to handle Volume object
 * @method Volume
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @param {K3D}
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config) {
        config.samples = config.samples || 512.0;
        config.gradient_step = config.gradient_step || 0.005;
        config.interpolation = typeof (config.interpolation) !== 'undefined' ? config.interpolation : true;

        const randomMul = typeof (window.randomMul) !== 'undefined' ? window.randomMul : 255.0;
        const geometry = new THREE.BoxBufferGeometry(1, 1, 1);
        const modelMatrix = new THREE.Matrix4();
        const translation = new THREE.Vector3();
        const rotation = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        const colorMap = (config.color_map && config.color_map.data) || null;
        let mask = null;
        let maskEnabled = false;
        let opacityFunction = (config.opacity_function && config.opacity_function.data) || null;
        const colorRange = config.color_range;
        const {samples} = config;
        let jitterTexture;

        if (opacityFunction === null) {
            opacityFunction = [colorMap[0], 0.0, colorMap[colorMap.length - 4], 1.0];
        }

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        modelMatrix.decompose(translation, rotation, scale);

        const texture = new THREE.Data3DTexture(
            config.volume.data,
            config.volume.shape[2],
            config.volume.shape[1],
            config.volume.shape[0],
        );

        texture.format = THREE.RedFormat;
        texture.type = typedArrayToThree(config.volume.data.constructor);

        texture.generateMipmaps = false;

        if (config.interpolation) {
            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;
        } else {
            texture.minFilter = THREE.NearestFilter;
            texture.magFilter = THREE.NearestFilter;
        }

        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.wrapR = THREE.ClampToEdgeWrapping;
        texture.needsUpdate = true;

        jitterTexture = new THREE.DataTexture(
            new Uint8Array(_.range(64 * 64).map(() => randomMul * Math.random())),
            64,
            64,
            THREE.RedFormat,
            THREE.UnsignedByteType,
        );
        jitterTexture.minFilter = THREE.LinearFilter;
        jitterTexture.magFilter = THREE.LinearFilter;
        jitterTexture.wrapS = THREE.MirroredRepeatWrapping;
        jitterTexture.wrapT = THREE.MirroredRepeatWrapping;
        jitterTexture.generateMipmaps = false;
        jitterTexture.needsUpdate = true;

        const canvas = colorMapHelper.createCanvasGradient(colorMap, 1024, 1, opacityFunction);
        const colormap = new THREE.CanvasTexture(
            canvas,
            THREE.UVMapping,
            THREE.ClampToEdgeWrapping,
            THREE.ClampToEdgeWrapping,
            THREE.NearestFilter,
            THREE.NearestFilter,
        );
        colormap.needsUpdate = true;

        if (config.mask.data.length > 0 && config.mask_opacities.data.length > 0) {
            mask = new THREE.Data3DTexture(
                config.mask.data,
                config.mask.shape[2],
                config.mask.shape[1],
                config.mask.shape[0],
            );
            mask.format = THREE.RedFormat;
            mask.type = THREE.UnsignedByteType;

            mask.generateMipmaps = false;
            mask.minFilter = THREE.NearestFilter;
            mask.magFilter = THREE.NearestFilter;
            mask.wrapS = THREE.ClampToEdgeWrapping;
            mask.wrapT = THREE.ClampToEdgeWrapping;
            mask.needsUpdate = true;

            maskEnabled = true;
        }

        const uniforms = {
            maskOpacities: {value: ensure256size(config.mask_opacities.data)},
            volumeMapSize: {
                value: new THREE.Vector3(
                    config.volume.shape[2],
                    config.volume.shape[1],
                    config.volume.shape[0],
                ),
            },
            low: {value: colorRange[0]},
            high: {value: colorRange[1]},
            gradient_step: {value: config.gradient_step},
            samples: {value: samples},
            translation: {value: translation},
            rotation: {value: rotation},
            scale: {value: scale},
            volumeTexture: {type: 't', value: texture},
            mask: {type: 't', value: mask},
            colormap: {type: 't', value: colormap},
            jitterTexture: {type: 't', value: jitterTexture},
        };

        const material = new THREE.ShaderMaterial({
            uniforms: _.merge(
                uniforms,
                THREE.UniformsLib.lights,
            ),
            defines: {
                USE_SPECULAR: 1,
                USE_MASK: (maskEnabled ? 1 : 0),
            },
            vertexShader: require('./shaders/MIP.vertex.glsl'),
            fragmentShader: require('./shaders/MIP.fragment.glsl'),
            side: THREE.BackSide,
            depthTest: false,
            depthWrite: false,
            lights: true,
            clipping: true,
            transparent: true,
        });

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        const object = new THREE.Mesh(geometry, material);
        object.applyMatrix4(modelMatrix);
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

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        if (typeof (changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
            obj.material.uniforms.low.value = changes.color_range[0];
            obj.material.uniforms.high.value = changes.color_range[1];

            resolvedChanges.color_range = null;
        }

        if (typeof (changes.volume) !== 'undefined' && !changes.volume.timeSeries) {
            if (obj.material.uniforms.volumeTexture.value.image.data.constructor === changes.volume.data.constructor) {
                obj.material.uniforms.volumeTexture.value.image.data = changes.volume.data;
                obj.material.uniforms.volumeTexture.value.needsUpdate = true;

                resolvedChanges.volume = null;
            }
        }

        if ((typeof (changes.color_map) !== 'undefined' && !changes.color_map.timeSeries)
            || (typeof (changes.opacity_function) !== 'undefined' && !changes.opacity_function.timeSeries)) {
            const canvas = colorMapHelper.createCanvasGradient(
                (changes.color_map && changes.color_map.data) || config.color_map.data,
                1024,
                1,
                (changes.opacity_function && changes.opacity_function.data) || config.opacity_function.data,
            );

            obj.material.uniforms.colormap.value.image = canvas;
            obj.material.uniforms.colormap.value.needsUpdate = true;

            resolvedChanges.color_map = null;
            resolvedChanges.opacity_function = null;
        }

        if (typeof (changes.mask) !== 'undefined' && !changes.mask.timeSeries) {
            if (obj.material.uniforms.mask.value !== null) {
                if (obj.material.uniforms.mask.value.image.data.constructor === changes.mask.data.constructor
                    && obj.material.uniforms.mask.value.image.width === changes.mask.shape[2]
                    && obj.material.uniforms.mask.value.image.height === changes.mask.shape[1]
                    && obj.material.uniforms.mask.value.image.depth === changes.mask.shape[0]) {
                    obj.material.uniforms.mask.value.image.data = changes.mask.data;
                    obj.material.uniforms.mask.value.needsUpdate = true;

                    resolvedChanges.mask = null;
                }
            }
        }

        if (typeof (changes.mask_opacities) !== 'undefined' && !changes.mask_opacities.timeSeries) {
            if (obj.material.uniforms.maskOpacities.value !== null) {
                obj.material.uniforms.maskOpacities.value = ensure256size(changes.mask_opacities.data);
                obj.material.uniforms.maskOpacities.value.needsUpdate = true;

                resolvedChanges.mask_opacities = null;
            }
        }

        ['samples', 'gradient_step'].forEach((key) => {
            if (changes[key] && !changes[key].timeSeries) {
                obj.material.uniforms[key].value = changes[key];
                resolvedChanges[key] = null;
            }
        });

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj});
        }
        return false;
    },
};
