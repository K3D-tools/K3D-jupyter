const THREE = require('three');
const interactionsHelper = require('../helpers/Interactions');
const colorMapHelper = require('../../../core/lib/helpers/colorMap');
const { typedArrayToThree } = require('../helpers/Fn');
const { areAllChangesResolve } = require('../helpers/Fn');
const { commonUpdate } = require('../helpers/Fn');
const { getSide } = require('../helpers/Fn');

/**
 * Loader strategy to handle Mesh object
 * @method Mesh
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config, K3D) {
        config.color = typeof (config.color) !== 'undefined' ? config.color : 255;
        config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;

        const modelMatrix = new THREE.Matrix4();
        const colorRange = config.color_range;
        const colorMap = (config.color_map && config.color_map.data) || null;
        const vertices = (config.vertices && config.vertices.data) || null;
        const indices = (config.indices && config.indices.data) || null;
        let opacityFunction = null;
        const geometry = new THREE.BufferGeometry();

        if (config.opacity_function && config.opacity_function.data && config.opacity_function.data.length > 0) {
            opacityFunction = config.opacity_function.data;
        }

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

        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));

        const texture = new THREE.Data3DTexture(
            config.volume.data,
            config.volume.shape[2],
            config.volume.shape[1],
            config.volume.shape[0],
        );
        texture.format = THREE.RedFormat;
        texture.type = typedArrayToThree(config.volume.data.constructor);

        texture.generateMipmaps = false;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.needsUpdate = true;

        const material = new THREE.ShaderMaterial({
            uniforms: {
                opacity: { value: config.opacity },
                low: { value: colorRange[0] },
                high: { value: colorRange[1] },
                volumeTexture: { type: 't', value: texture },
                colormap: { type: 't', value: colormap },
                b1: {
                    type: 'v3',
                    value: new THREE.Vector3(
                        config.volume_bounds.data[0],
                        config.volume_bounds.data[2],
                        config.volume_bounds.data[4],
                    ),
                },
                b2: {
                    type: 'v3',
                    value: new THREE.Vector3(
                        config.volume_bounds.data[1],
                        config.volume_bounds.data[3],
                        config.volume_bounds.data[5],
                    ),
                },
            },
            side: getSide(config),
            vertexShader: require('./shaders/MeshVolume.vertex.glsl'),
            fragmentShader: require('./shaders/MeshVolume.fragment.glsl'),
            depthWrite: (config.opacity === 1.0 && opacityFunction === null),
            transparent: (config.opacity !== 1.0 || opacityFunction !== null),
            lights: false,
            clipping: true,
        });

        if (config.flat_shading === false) {
            geometry.computeVertexNormals();
        }

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        const object = new THREE.Mesh(geometry, material);

        interactionsHelper.init(config, object, K3D);

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix4(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        interactionsHelper.update(config, changes, resolvedChanges, obj);

        if (typeof (changes.volume) !== 'undefined' && !changes.volume.timeSeries) {
            if (obj.material.uniforms.volumeTexture.value.image.data.constructor === changes.volume.data.constructor
                && obj.material.uniforms.volumeTexture.value.image.width === changes.volume.shape[2]
                && obj.material.uniforms.volumeTexture.value.image.height === changes.volume.shape[1]
                && obj.material.uniforms.volumeTexture.value.image.depth === changes.volume.shape[0]) {
                obj.material.uniforms.volumeTexture.value.image.data = changes.volume.data;
                obj.material.uniforms.volumeTexture.value.needsUpdate = true;

                resolvedChanges.volume = null;
            }
        }

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

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};
