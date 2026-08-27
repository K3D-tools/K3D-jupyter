const THREE = require('three');
const BufferGeometryUtils = require('three/examples/jsm/utils/BufferGeometryUtils');
const { areAllChangesResolve } = require('../helpers/Fn');
const { commonUpdate } = require('../helpers/Fn');

/**
 * Loader strategy to handle STL object
 * @method STL
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config, K3D) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 255;
        config.wireframe = typeof (config.wireframe) !== 'undefined' ? config.wireframe : false;
        config.flat_shading = typeof (config.flat_shading) !== 'undefined' ? config.flat_shading : true;
        config.roughness = typeof (config.roughness) !== 'undefined' ? config.roughness : 0.4;
        config.metalness = typeof (config.metalness) !== 'undefined' ? config.metalness : 0.0;

        const loader = new THREE.STLLoader();
        const modelMatrix = new THREE.Matrix4();
        const MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshStandardMaterial;
        // MeshBasicMaterial warns about every lighting parameter, so the wireframe branch gets none
        let material = new MaterialConstructor(config.wireframe ? {
            color: config.color,
            side: THREE.DoubleSide,
            wireframe: true,
        } : {
            color: config.color,
            emissive: 0,
            roughness: config.roughness,
            metalness: config.metalness,
            flatShading: config.flat_shading,
            side: THREE.DoubleSide,
            wireframe: false,
        });
        const { text } = config;
        const { binary } = config;
        let geometry;

        if (text === null || typeof (text) === 'undefined') {
            geometry = loader.parse(binary.data.buffer);
        } else {
            geometry = loader.parse(text);
        }

        if (geometry.hasColors) {
            material = new THREE.MeshStandardMaterial({
                opacity: geometry.alpha,
                roughness: config.roughness,
                metalness: config.metalness,
                vertexColors: true,
                wireframe: config.wireframe,
            });
        }

        if (config.flat_shading === false) {
            geometry = BufferGeometryUtils.mergeVertices(geometry);
            geometry.computeVertexNormals();
        }

        const object = new THREE.Mesh(geometry, material);

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix4(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        if (typeof (changes.color) !== 'undefined' && !changes.color.timeSeries
            && obj.material && obj.material.color) {
            obj.material.color.set(changes.color);

            resolvedChanges.color = null;
        }

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};
