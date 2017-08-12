'use strict';

/**
 * Loader strategy to handle Surface object
 * @method MarchingCubes
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var heights = config.heights.buffer,
        width = config.heights.shape[1],
        height = config.heights.shape[0],
        modelMatrix = new THREE.Matrix4(),
        material = new THREE.MeshPhongMaterial({
            color: config.color,
            emissive: 0,
            shininess: 25,
            specular: 0x111111,
            side: THREE.DoubleSide,
            shading: THREE.FlatShading
        }),
        geometry = new THREE.BufferGeometry(),
        vertices = new Float32Array((width - 1) * (height - 1) * 3 * 3 * 2),
        object,
        x, y, i, p;

    for (y = 0, i = 0, p = 0; y < height - 1; y++) {
        for (x = 0; x < width - 1; x++, p++, i += 18) {
            // Performance over readability
            vertices[i] = vertices[i + 9] = x / (width - 1);
            vertices[i + 1] = vertices[i + 10] = y / (height - 1);
            vertices[i + 2] = vertices[i + 11] = heights[p];

            vertices[i + 3] = vertices[i + 15] = (x + 1) / (width - 1);
            vertices[i + 4] = vertices[i + 16] = (y + 1) / (height - 1);
            vertices[i + 5] = vertices[i + 17] = heights[p + 1 + width];

            vertices[i + 6] = (x + 1) / (width - 1);
            vertices[i + 7] = y / (height - 1);
            vertices[i + 8] = heights[p + 1];

            vertices[i + 12] = x / (width - 1);
            vertices[i + 13] = (y + 1) / (height - 1);
            vertices[i + 14] = heights[p + width];
        }

        //skip last column
        p++;
    }

    geometry.addAttribute('position', new THREE.BufferAttribute(vertices, 3));

    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    object = new THREE.Mesh(geometry, material);
    modelMatrix.set.apply(modelMatrix, config.model_matrix.buffer);

    object.position.set(-0.5, -0.5, 0);
    object.updateMatrix();

    object.applyMatrix(modelMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
