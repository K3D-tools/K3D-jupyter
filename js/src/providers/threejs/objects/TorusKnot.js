'use strict';
/**
 * Loader strategy to handle TorusKnot object
 * @method TorusKnot
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {K3D.Config} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var object = new THREE.Object3D(),
        modelViewMatrix = new THREE.Matrix4();

    function updateGroupGeometry(mesh, geometry) {
        mesh.children[0].geometry.dispose();
        mesh.children[1].geometry.dispose();

        mesh.children[0].geometry = new THREE.WireframeGeometry(geometry);
        mesh.children[1].geometry = geometry;

        mesh.children[0].geometry.computeBoundingSphere();
        mesh.children[1].geometry.computeBoundingSphere();
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
            color: config.get('color'),
            emissive: 0x072534,
            side: THREE.DoubleSide,
            shading: THREE.FlatShading
        })
    ));

    updateGroupGeometry(object,
        new THREE.TorusKnotGeometry(
            config.get('radius'),
            config.get('tube'),
            64,
            config.get('knotsNumber'),
            2, 3, 1)
    );

    modelViewMatrix.set.apply(modelViewMatrix, config.get('modelViewMatrix'));
    object.applyMatrix(modelViewMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
