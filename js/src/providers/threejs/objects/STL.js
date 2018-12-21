'use strict';

/**
 * Loader strategy to handle STL object
 * @method STL
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config) {
        config.visible = typeof(config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof(config.color) !== 'undefined' ? config.color : 255;
        config.wireframe = typeof(config.wireframe) !== 'undefined' ? config.wireframe : false;
        config.flat_shading = typeof(config.flat_shading) !== 'undefined' ? config.flat_shading : true;

        var loader = new THREE.STLLoader(),
            modelMatrix = new THREE.Matrix4(),
            MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial,
            material = new MaterialConstructor({
                color: config.color,
                emissive: 0,
                shininess: 50,
                specular: 0x111111,
                flatShading: config.flat_shading,
                side: THREE.DoubleSide,
                wireframe: config.wireframe
            }),
            text = config.text,
            binary = config.binary,
            geometry,
            object;

        if (text === null || typeof(text) === 'undefined') {
            if (typeof(binary.buffer) !== 'undefined') {
                geometry = loader.parse(binary.buffer);
            } else {
                geometry = loader.parse(binary);
            }
        } else {
            geometry = loader.parse(text);
        }

        if (geometry.hasColors) {
            material = new THREE.MeshPhongMaterial({
                opacity: geometry.alpha,
                vertexColors: THREE.VertexColors,
                wireframe: config.wireframe
            });
        }

        if (config.flat_shading === false) {
            var geo = new THREE.Geometry().fromBufferGeometry(geometry);
            geo.mergeVertices();
            geo.computeVertexNormals();
            geometry.fromGeometry(geo);
        }

        object = new THREE.Mesh(geometry, material);

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    }
};
