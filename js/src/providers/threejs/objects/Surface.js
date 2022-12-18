const THREE = require('three');
const interactionsHelper = require('../helpers/Interactions');
const {handleColorMap} = require('../helpers/Fn');
const {areAllChangesResolve} = require('../helpers/Fn');
const {commonUpdate} = require('../helpers/Fn');

/**
 * Loader strategy to handle Surface object
 * @method MarchingCubes
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

        const heights = config.heights.data;
        const width = config.heights.shape[1];
        const height = config.heights.shape[0];
        const modelMatrix = new THREE.Matrix4();
        const MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial;
        const colorRange = config.color_range;
        const colorMap = (config.color_map && config.color_map.data) || null;
        const attribute = (config.attribute && config.attribute.data) || null;
        const material = new MaterialConstructor({
            color: config.color,
            emissive: 0,
            shininess: 50,
            specular: 0x111111,
            side: THREE.DoubleSide,
            flatShading: config.flat_shading,
            wireframe: config.wireframe,
        });
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array(width * height * 3);
        const indices = [];
        let x;
        let y;
        let i;
        let p;

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
                indices[i + 1] = x + 1 + y * width;
                indices[i + 3] = indices[i + 1];
                indices[i + 2] = x + (y + 1) * width;
                indices[i + 5] = indices[i + 2];
                indices[i + 4] = x + 1 + (y + 1) * width;
            }
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(indices);

        if (config.flat_shading === false) {
            geometry.computeVertexNormals();
        }

        if (attribute && colorRange && colorMap && attribute.length > 0
            && colorRange.length > 0 && colorMap.length > 0) {
            handleColorMap(geometry, colorMap, colorRange, attribute, material);
        }

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        const object = new THREE.Mesh(geometry, material);

        interactionsHelper.init(config, object, K3D);

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);

        object.position.set(-0.5, -0.5, 0);
        object.initialPosition = object.position.clone();
        object.updateMatrix();

        object.applyMatrix4(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        interactionsHelper.update(config, changes, resolvedChanges, obj);
        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj});
        }
        return false;
    },
};
