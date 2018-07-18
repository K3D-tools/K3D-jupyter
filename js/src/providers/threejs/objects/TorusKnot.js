'use strict';
/**
 * Loader strategy to handle TorusKnot object
 * @method TorusKnot
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var object = new THREE.Object3D(),
        modelMatrix = new THREE.Matrix4();

    function updateGroupGeometry(mesh, geometry) {
        mesh.children[0].geometry.dispose();
        mesh.children[1].geometry.dispose();

        mesh.children[0].geometry = new THREE.WireframeGeometry(geometry);
        mesh.children[1].geometry = geometry;

        mesh.children[0].geometry.computeBoundingSphere();
        mesh.children[1].geometry.computeBoundingSphere();

        mesh.children[0].geometry.computeBoundingBox();
        mesh.children[1].geometry.computeBoundingBox();
    }

    object.add(new THREE.LineSegments(
        new THREE.Geometry(),
        new THREE.LineBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.5
        })
    ));

    object.add(new THREE.Mesh(
        new THREE.Geometry(),
        new THREE.MeshPhongMaterial({
            color: config.color,
            emissive: 0,
            side: THREE.DoubleSide,
            flatShading: true
        })
    ));

    updateGroupGeometry(object,
        new THREE.TorusKnotGeometry(
            config.radius,
            config.tube,
            64,
            config.knotsNumber,
            2, 3)
    );

    modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
    object.applyMatrix(modelMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
