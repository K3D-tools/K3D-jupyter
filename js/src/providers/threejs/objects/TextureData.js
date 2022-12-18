const THREE = require('three');
const interactionsHelper = require('../helpers/Interactions');
const colorMapHelper = require('../../../core/lib/helpers/colorMap');
const {areAllChangesResolve} = require('../helpers/Fn');
const {commonUpdate} = require('../helpers/Fn');
const {typedArrayToThree} = require('../helpers/Fn');

/**
 * Loader strategy to handle Texture object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config, K3D) {
        return new Promise((resolve) => {
            const geometry = new THREE.PlaneBufferGeometry(1, 1);
            const modelMatrix = new THREE.Matrix4();
            const colorMap = (config.color_map && config.color_map.data) || null;
            let opacityFunction = (config.opacity_function && config.opacity_function.data) || null;
            const colorRange = config.color_range;

            config.interpolation = typeof (config.interpolation) !== 'undefined' ? config.interpolation : true;

            if (opacityFunction === null || opacityFunction.length === 0) {
                opacityFunction = [colorMap[0], 1.0, colorMap[colorMap.length - 4], 1.0];

                config.opacity_function = {
                    data: opacityFunction,
                    shape: [4],
                };
            }

            const texture = new THREE.DataTexture(
                config.attribute.data,
                config.attribute.shape[1],
                config.attribute.shape[0],
                THREE.RedFormat,
                typedArrayToThree(config.attribute.data.constructor),
            );

            if (config.interpolation) {
                texture.minFilter = THREE.LinearFilter;
                texture.magFilter = THREE.LinearFilter;
                texture.anisotropy = K3D.getWorld().renderer.capabilities.getMaxAnisotropy();
            } else {
                texture.minFilter = THREE.NearestFilter;
                texture.magFilter = THREE.NearestFilter;
                texture.anisotropy = 0;
            }

            texture.generateMipmaps = false;

            texture.needsUpdate = true;

            const canvas = colorMapHelper.createCanvasGradient(colorMap, 1024, opacityFunction);
            const colormap = new THREE.CanvasTexture(
                canvas,
                THREE.UVMapping,
                THREE.ClampToEdgeWrapping,
                THREE.ClampToEdgeWrapping,
                THREE.NearestFilter,
                THREE.NearestFilter,
            );
            colormap.needsUpdate = true;

            const uniforms = {
                low: {value: colorRange[0]},
                high: {value: colorRange[1]},
                map: {type: 't', value: texture},
                colormap: {type: 't', value: colormap},
            };

            const material = new THREE.ShaderMaterial({
                uniforms,
                vertexShader: require('./shaders/Texture.vertex.glsl'),
                fragmentShader: require('./shaders/Texture.fragment.glsl'),
                side: THREE.DoubleSide,
                clipping: true,
            });

            if (config.puv.data.length === 9) {
                const positionArray = geometry.attributes.position.array;

                const p = new THREE.Vector3().fromArray(config.puv.data, 0);
                const u = new THREE.Vector3().fromArray(config.puv.data, 3);
                const v = new THREE.Vector3().fromArray(config.puv.data, 6);

                p.toArray(positionArray, 0);
                p.clone().add(u).toArray(positionArray, 3);
                p.clone().add(v).toArray(positionArray, 6);
                p.clone().add(v).add(u).toArray(positionArray, 9);

                geometry.computeVertexNormals();
            }

            geometry.computeBoundingSphere();
            geometry.computeBoundingBox();

            const object = new THREE.Mesh(geometry, material);

            interactionsHelper.init(config, object, K3D);

            modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
            object.applyMatrix4(modelMatrix);
            object.updateMatrixWorld();

            object.onRemove = function () {
                object.material.uniforms.map.value.dispose();
                object.material.uniforms.map.value = undefined;
                object.material.uniforms.colormap.value.dispose();
                object.material.uniforms.colormap.value = undefined;
            };

            resolve(object);
        });
    },

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        interactionsHelper.update(config, changes, resolvedChanges, obj);

        if (typeof (changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
            obj.material.uniforms.low.value = changes.color_range[0];
            obj.material.uniforms.high.value = changes.color_range[1];

            resolvedChanges.color_range = null;
        }

        if ((typeof (changes.color_map) !== 'undefined' && !changes.color_map.timeSeries)
            || (typeof (changes.opacity_function) !== 'undefined' && !changes.opacity_function.timeSeries)) {
            if (!(changes.opacity_function && obj.material.transparent === false)) {
                const canvas = colorMapHelper.createCanvasGradient(
                    (changes.color_map && changes.color_map.data) || config.color_map.data,
                    1024,
                    (changes.opacity_function && changes.opacity_function.data) || config.opacity_function.data,
                );

                obj.material.uniforms.colormap.value.image = canvas;
                obj.material.uniforms.colormap.value.needsUpdate = true;

                resolvedChanges.color_map = null;
                resolvedChanges.opacity_function = null;
            }
        }

        if (typeof (changes.attribute) !== 'undefined' && !changes.attribute.timeSeries) {
            if (obj.material.uniforms.map.value.image.data.constructor === changes.attribute.data.constructor
                && obj.material.uniforms.map.value.image.width === changes.attribute.shape[1]
                && obj.material.uniforms.map.value.image.height === changes.attribute.shape[0]) {
                obj.material.uniforms.map.value.image.data = changes.attribute.data;
                obj.material.uniforms.map.value.needsUpdate = true;

                resolvedChanges.attribute = null;
            }
        }

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj});
        }
        return false;
    },
};
