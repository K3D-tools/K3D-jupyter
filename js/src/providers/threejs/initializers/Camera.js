'use strict';

var THREE = require('three');

/**
 * Camera initializer for Three.js library
 * @this K3D.Core~world
 * @method Canvas
 * @memberof K3D.Providers.ThreeJS.Initializers
 */
module.exports = function (K3D) {
    var currentFar = 1000;

    this.camera = new THREE.PerspectiveCamera(K3D.parameters.camera_fov, this.width / this.height, 0.1, currentFar);
    this.camera.position.set(2, -3, 0.2);
    this.camera.up.set(0, 0, 1);


    this.axesHelper.camera = new THREE.PerspectiveCamera(K3D.parameters.camera_fov,
        this.axesHelper.width / this.axesHelper.height, 0.1, 1000);
    this.axesHelper.camera.position.set(2, 0.5, 0.5);
    this.axesHelper.camera.lookAt(0.5, 0.5, 0.5);
    this.axesHelper.camera.up.copy(this.camera.up);

    this.setupCamera = function (array, fov) {
        if (fov) {
            this.camera.fov = this.axesHelper.camera.fov = fov;
            this.camera.updateProjectionMatrix();
            this.axesHelper.camera.updateProjectionMatrix();
        }

        if (array) {
            this.controls.object.position.fromArray(array);

            if (array.length === 9) {
                this.controls.object.up.fromArray(array, 6);
            }

            this.controls.target.fromArray(array, 3);
            this.controls.update();
        }
    };

    this.setCameraToFitScene = function (force) {
        var camDistance,
            sceneBoundingBox = new THREE.Box3().setFromArray(K3D.parameters.grid),
            sceneBoundingSphere;

        if (!K3D.parameters.cameraAutoFit && !force) {
            return;
        }

        if (this.K3DObjects.children.length > 0) {
            sceneBoundingBox = K3D.getSceneBoundingBox() || sceneBoundingBox;
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
         |                       ....`                           /       /             ..     |         sceneBoundingSphere.radius
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

        sceneBoundingSphere = sceneBoundingBox.getBoundingSphere(new THREE.Sphere());

        camDistance = sceneBoundingSphere.radius * 1.5 / Math.sin(THREE.Math.degToRad(K3D.parameters.camera_fov / 2.0));

        this.camera.position.subVectors(
            sceneBoundingSphere.center,
            this.camera.getWorldDirection(new THREE.Vector3()).setLength(camDistance)
        );
        this.controls.target = sceneBoundingSphere.center;
    };
};
