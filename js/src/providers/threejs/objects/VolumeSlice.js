const THREE = require('three');
const interactionsVolumeSlice = require('../interactions/VolumeSlice');
const interactionsHelper = require('../helpers/Interactions');
const colorMapHelper = require('../../../core/lib/helpers/colorMap');
const _ = require("../../../lodash");
const typedArrayToThree = require('../helpers/Fn').typedArrayToThree;
const areAllChangesResolve = require('../helpers/Fn').areAllChangesResolve;
const commonUpdate = require('../helpers/Fn').commonUpdate;

const normals = [
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,

    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,

    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
];

function getTexture(data) {
    const activeMasks = colorMapHelper.createCanvasColorList(data);

    return new THREE.CanvasTexture(
        activeMasks,
        THREE.UVMapping,
        THREE.ClampToEdgeWrapping,
        THREE.ClampToEdgeWrapping,
        THREE.NearestFilter,
        THREE.NearestFilter,
    );
}

function getPositions(slice, shape) {
    const ret = new Float32Array(3 * 3 * 4).fill(0);
    const deltaX = (slice[0] + 0.5) / shape[2];
    const deltaY = (slice[1] + 0.5) / shape[1];
    const deltaZ = (slice[2] + 0.5) / shape[0];

    if (deltaX >= 0) {
        ret.set([
            -0.5 + deltaX, -0.5, -0.5,
            -0.5 + deltaX, 0.5, -0.5,
            -0.5 + deltaX, 0.5, 0.5,
            -0.5 + deltaX, -0.5, 0.5,
        ], 0);
    }

    if (deltaY >= 0) {
        ret.set([
            -0.5, -0.5 + deltaY, -0.5,
            0.5, -0.5 + deltaY, -0.5,
            0.5, -0.5 + deltaY, 0.5,
            -0.5, -0.5 + deltaY, 0.5,
        ], 3 * 4);
    }

    if (deltaZ >= 0) {
        ret.set([
            -0.5, -0.5, -0.5 + deltaZ,
            0.5, -0.5, -0.5 + deltaZ,
            0.5, 0.5, -0.5 + deltaZ,
            -0.5, 0.5, -0.5 + deltaZ,
        ], 2 * 3 * 4);
    }

    return ret;
}

function addTextureToUniforms(uniforms, config) {
    let d = config.volume.reduce(function (ret, volume, id) {
        const texture = new THREE.Data3DTexture(
            volume.data,
            volume.shape[2],
            volume.shape[1],
            volume.shape[0],
        );
        texture.format = THREE.RedFormat;
        texture.type = typedArrayToThree(volume.data.constructor);

        texture.generateMipmaps = false;
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;

        if (config.interpolation > 0) {
            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;
        } else {
            texture.minFilter = THREE.NearestFilter;
            texture.magFilter = THREE.NearestFilter;
        }

        texture.needsUpdate = true;

        ret.low.push(config.color_range[id * 2]);
        ret.high.push(config.color_range[id * 2 + 1]);
        ret.volumeTexture.push(texture);
        ret.volumeSize.push(new THREE.Vector3(volume.shape[2], volume.shape[1], volume.shape[0]));

        return ret;

    }, {low: [], high: [], volumeTexture: [], volumeSize: []});

    uniforms['low'] = {value: d.low};
    uniforms['high'] = {value: d.high};
    uniforms['volumeTexture'] = {type: 'tv', value: d.volumeTexture};
    uniforms['volumeSize'] = {value: d.volumeSize};
}

/**
 * Loader strategy to handle Volume Slice object
 * @method VolumeSlice
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config, K3D) {
        config.color = typeof (config.color) !== 'undefined' ? config.color : 255;
        config.mask_opacity = typeof (config.mask_opacity) !== 'undefined' ? config.mask_opacity : 0.5;
        config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;
        config.interpolation = typeof (config.interpolation) !== 'undefined' ? config.interpolation : 1;

        const modelMatrix = new THREE.Matrix4();
        const colorMap = (config.color_map && config.color_map.data) || null;
        let opacityFunction = null;
        const geometry = new THREE.BufferGeometry();
        let mask = [];
        let maskColors = [];
        let activeMasks = [];
        let activeMasksCount = 0;

        if (config.opacity_function && config.opacity_function.data && config.opacity_function.data.length > 0) {
            opacityFunction = config.opacity_function.data;
        }

        const shape = Array.isArray(config.volume) ? config.volume[0].shape : config.volume.shape;
        config.volume = Array.isArray(config.volume) ? config.volume : [config.volume];

        const canvas = colorMapHelper.createCanvasGradient(colorMap, 1024, config.volume.length, opacityFunction);
        const colormap = new THREE.CanvasTexture(
            canvas,
            THREE.UVMapping,
            THREE.ClampToEdgeWrapping,
            THREE.ClampToEdgeWrapping,
            THREE.NearestFilter,
            THREE.NearestFilter,
        );

        colormap.needsUpdate = true;

        geometry.setAttribute(
            'position',
            new THREE.BufferAttribute(
                getPositions(
                    [config.slice_x, config.slice_y, config.slice_z],
                    shape,
                ),
                3,
            ),
        );
        geometry.setAttribute(
            'normals',
            new THREE.BufferAttribute(new Float32Array(normals), 3),
        );

        geometry.setIndex(new THREE.BufferAttribute(new Uint32Array([
            0, 1, 2, 0, 2, 3,
            4, 5, 6, 4, 6, 7,
            8, 9, 10, 8, 10, 11,
        ]), 1));

        if (config.mask.data.length > 0 && config.color_map_masks.data.length > 0) {
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

            activeMasks = getTexture(config.active_masks.data);
            maskColors = getTexture(config.color_map_masks.data);

            activeMasksCount = config.active_masks.data.length;
        }

        let uniforms = {
            opacity: {value: config.opacity},
            mask: {type: 't', value: mask},
            maskColors: {type: 't', value: maskColors},
            maskOpacity: {value: config.mask_opacity},
            activeMasksCount: {value: activeMasksCount},
            activeMasks: {type: 't', value: activeMasks},
            colormap: {type: 't', value: colormap},
        };

        addTextureToUniforms(uniforms, config);

        const material = new THREE.ShaderMaterial({
            uniforms: _.merge(
                uniforms,
                THREE.UniformsLib.lights,
            ),
            defines: {
                CUBIC: config.interpolation === 2 ? 1 : 0
            },
            side: THREE.DoubleSide,
            vertexShader: require('./shaders/VolumeSlice.vertex.glsl'),
            fragmentShader: require('./shaders/VolumeSlice.fragment.glsl'),
            depthWrite: (config.opacity === 1.0 && opacityFunction === null),
            transparent: (config.opacity !== 1.0 || opacityFunction !== null),
            lights: false,
            clipping: true,
            onBeforeCompile: function (shader) {
                shader.fragmentShader = shader.fragmentShader.replace(/TEXTURE_COUNT/g, config.volume.length);
                shader.vertexShader = shader.vertexShader.replace(/TEXTURE_COUNT/g, config.volume.length);
            }
        });

        if (config.flat_shading === false) {
            geometry.computeVertexNormals();
        }

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        const object = new THREE.Mesh(geometry, material);

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix4(modelMatrix);

        object.updateMatrixWorld();

        interactionsHelper.init(config, object, K3D, interactionsVolumeSlice);

        return Promise.resolve(object);
    },

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        const shape = Array.isArray(config.volume) ? config.volume[0].shape : config.volume.shape;

        if (typeof (changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
            if (Array.isArray(obj.material.uniforms.low.value)) {
                obj.material.uniforms.low.value[0] = changes.color_range[0];
                obj.material.uniforms.high.value[0] = changes.color_range[1];
            } else {
                obj.material.uniforms.low.value = changes.color_range[0];
                obj.material.uniforms.high.value = changes.color_range[1];
            }

            resolvedChanges.color_range = null;
        }

        let slice_position_updated = false;
        ['x', 'y', 'z'].forEach((axis) => {
            if (!slice_position_updated && typeof (changes[`slice_${axis}`]) !== 'undefined' && !changes[`slice_${axis}`].timeSeries) {
                const data = obj.geometry.attributes.position.array;
                const newData = getPositions(
                    [
                        typeof (changes.slice_x) !== 'undefined' ? changes.slice_x : config.slice_x,
                        typeof (changes.slice_y) !== 'undefined' ? changes.slice_y : config.slice_y,
                        typeof (changes.slice_z) !== 'undefined' ? changes.slice_z : config.slice_z,
                    ],
                    shape,
                );

                for (let i = 0; i < data.length; i++) {
                    data[i] = newData[i];
                }

                obj.geometry.attributes.position.needsUpdate = true;

                resolvedChanges.slice_x = null;
                resolvedChanges.slice_y = null;
                resolvedChanges.slice_z = null;
                slice_position_updated = true;
            }
        });

        if (typeof (changes.volume) !== 'undefined' && !changes.volume.timeSeries) {
            if (!Array.isArray(changes.volume)) {
                changes.volume = [changes.volume];
            }

            if (changes.volume.length === obj.material.uniforms.volumeTexture.value.length) {
                let allDone = true;

                changes.volume.forEach(function (volume, i) {
                    let val = obj.material.uniforms.volumeTexture.value[i];

                    if (val.image.data.constructor === volume.data.constructor && val.image.width === volume.shape[2]
                        && val.image.height === volume.shape[1] && val.image.depth === volume.shape[0]) {
                        val.image.data = volume.data;
                        val.needsUpdate = true;
                    } else {
                        allDone = false;
                    }
                })

                if (allDone) {
                    resolvedChanges.volume = null;
                }
            }
        }

        if (typeof (changes.mask) !== 'undefined' && !changes.mask.timeSeries) {
            if (obj.material.uniforms.mask.value.image.data.length > 0
                && obj.material.uniforms.mask.value.image.data.constructor === changes.mask.data.constructor) {
                obj.material.uniforms.mask.value.image.data = changes.mask.data;
                obj.material.uniforms.mask.value.needsUpdate = true;

                resolvedChanges.mask = null;
            }
        }

        if (typeof (changes.active_masks) !== 'undefined' && !changes.active_masks.timeSeries) {
            obj.material.uniforms.activeMasks.value = getTexture(changes.active_masks.data);
            obj.material.uniforms.activeMasks.value.needsUpdate = true;
            obj.material.uniforms.activeMasksCount.value = changes.active_masks.data.length;

            resolvedChanges.active_masks = null;
        }

        if (typeof (changes.color_map_masks) !== 'undefined' && !changes.color_map_masks.timeSeries) {
            obj.material.uniforms.maskColors.value = getTexture(changes.color_map_masks.data);
            obj.material.uniforms.maskColors.value.needsUpdate = true;

            resolvedChanges.color_map_masks = null;
        }

        if (typeof (changes.mask_opacity) !== 'undefined' && !changes.mask_opacity.timeSeries) {
            obj.material.uniforms.maskOpacity.value = changes.mask_opacity;

            resolvedChanges.mask_opacity = null;
        }

        if ((typeof (changes.color_map) !== 'undefined' && !changes.color_map.timeSeries)
            || (typeof (changes.opacity_function) !== 'undefined' && !changes.opacity_function.timeSeries)) {
            const canvas = colorMapHelper.createCanvasGradient(
                (changes.color_map && changes.color_map.data) || config.color_map.data,
                1024,
                config.volume.length,
                (changes.opacity_function && changes.opacity_function.data) || config.opacity_function.data,
            );

            obj.material.uniforms.colormap.value.image = canvas;
            obj.material.uniforms.colormap.value.needsUpdate = true;

            resolvedChanges.color_map = null;
            resolvedChanges.opacity_function = null;
        }

        interactionsHelper.update(config, changes, resolvedChanges, obj, K3D);

        ['opacity'].forEach((key) => {
            if (changes[key] && !changes[key].timeSeries) {
                obj.material.uniforms[key].value = changes[key];
                obj.material.needsUpdate = true;

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
