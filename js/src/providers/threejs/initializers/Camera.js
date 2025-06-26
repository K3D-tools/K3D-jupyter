const THREE = require('three');
const {cameraModes} = require('../../../core/lib/cameraMode');
const {recalculateFrustum} = require('../helpers/Fn');

/**
 * Camera initializer for Three.js library
 * @this K3D.Core~world
 * @method Canvas
 * @memberof K3D.Providers.ThreeJS.Initializers
 */
module.exports = function (K3D) {
    const currentFar = 1000;

    this.camera = new THREE.PerspectiveCamera(K3D.parameters.cameraFov, this.width / this.height, 0.1, currentFar);
    this.camera.position.set(2, -3, 0.2);
    this.camera.up.set(0, 0, 1);
    this.camera.frustum = new THREE.Frustum();

    this.axesHelper.camera = new THREE.PerspectiveCamera(
        K3D.parameters.cameraFov,
        this.axesHelper.width / this.axesHelper.height,
        0.1,
        1000,
    );
    this.axesHelper.camera.position.set(2, 0.5, 0.5);
    this.axesHelper.camera.lookAt(0.5, 0.5, 0.5);
    this.axesHelper.camera.up.copy(this.camera.up);

    this.setupCamera = function (array, fov, silent) {
        if (fov) {
            if (typeof (this.camera.fov) !== 'undefined' && typeof (this.axesHelper.camera.fov) !== 'undefined') {
                this.camera.fov = fov;
                this.axesHelper.camera.fov = fov;
                this.controls.dispatchEvent({type: 'change'});
            }
        }

        if (array) {
            this.controls.object.position.fromArray(array);

            if (array.length === 9) {
                this.controls.object.up.fromArray(array, 6);
                this.axesHelper.camera.up.copy(this.controls.object.up);
            }

            this.controls.target.fromArray(array, 3);
        }

        this.camera.updateProjectionMatrix();
        this.axesHelper.camera.updateProjectionMatrix();

        recalculateFrustum(this.camera);

        this.controls.update(silent);
    };

    this.setCameraToFitScene = function (force, factor) {
        let sceneBoundingBox = new THREE.Box3().setFromArray(K3D.parameters.grid);

        if (K3D.parameters.cameraMode === cameraModes.sliceViewer) {
            return;
        }

        if (!K3D.parameters.cameraAutoFit && !force) {
            return;
        }

        sceneBoundingBox = K3D.getSceneBoundingBox() || sceneBoundingBox;

        if (typeof (factor) === 'undefined') {
            factor = 1.5;
        }

        // Compute the distance the camera should be to fit the entire bounding sphere

        /*
     |                                                                                       ........
     |                                                                                  .....
     |                                                                              .--.  |
     |                                                                         `:///-.....|........
     |                                                                    `-:--.          |        ..--`
     |                                                                ..://`              |            .--`
     |                                                            ...-:-`                 |               `--
     |                                                       .....  -o....................|.................-o-
     |                                                   ....     .- / .                  |                  /`:.
     |                                              ....`        :.  /   .                |    Height        /  .-
     |                                         .....            :    /    ..              |                  /   `:
     |                                     ....`               /     /      ..            |                  /    `:
     |                                ....`                   :      /        ..          |                  /     `:
     |                             ....                       ..      /          .. fov / 2|                  /      -`
     |                       ....`                           /       /             ..     |   sceneBoundingSphere.radius
     |                  .....                               `:       /               .    |              ....+.....:`/
     |     CAMERA   ....`     fov / 2                       -.       /                 .. |    ..........`   /       :`
     |          ...`                                        -`       /                   .|...`              /       -.
     |     []<``...` ----------------------------------------------------------------------                  /       :`
     |             .....                      camDistance    :       /                                       /       /
     |                  `.....                               /       /                                       /       :
     |                       `.....                          `:      /                                       /      :
     |                            `.....                      -`     /                                       /     .-
     |                                  ...-.                  :`    /                                       /    .-
     |                                      `.--:-.             :-   /                                       /   .-
     |                                            `.....         --  /                                       /  -.
     |                                                 `.....     `:./                                       /.:
     |                                                       .....  .+:.....................................:+`
     |                                                            ....::-                                `--`
     |                                                                 .-/:-`                         `--.
     |                                                                      .::::.               `....`
     |                                                                           .--://-.........`
     |                                                                                 .....
     |                                                                                      .....
     |                                                                                           .....

     Height / camDistance = tan(fov/2) <=>
     camDistance = Height / tan(fov.2)

     sceneBoundingSphere.radius / Height = cos(fov/2) <=>
     Height = sceneBoundingSphere.radius / cos(fov/2)

     camDistance = (sceneBoundingSphere.radius / cos(fov/2)) / tan(fov/2) <=>
     camDistance = sceneBoundingSphere.radius / sin(fov/2);
     */

        const sceneBoundingSphere = sceneBoundingBox.getBoundingSphere(new THREE.Sphere());

        const camDistance = (sceneBoundingSphere.radius * factor) / Math.sin(
            THREE.Math.degToRad(K3D.parameters.cameraFov / 2.0),
        );

        this.camera.position.subVectors(
            sceneBoundingSphere.center,
            this.camera.getWorldDirection(new THREE.Vector3()).setLength(camDistance),
        );

        if (K3D.parameters.cameraMode === cameraModes.fly) {
            this.controls.target = this.camera.position.clone().add(
                this.camera.getWorldDirection(new THREE.Vector3()).setLength(camDistance / 2.0),
            );
        } else {
            this.controls.target = sceneBoundingSphere.center;
        }

        this.controls.update();
    };
};
