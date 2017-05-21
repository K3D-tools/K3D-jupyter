'use strict';

var startingSceneBoundingBox =  require('./Scene').startingSceneBoundingBox;

/**
 * Camera initializer for Three.js library
 * @this K3D.Core~world
 * @method Canvas
 * @memberof K3D.Providers.ThreeJS.Initializers
 */
module.exports = function (K3D) {
    var fov = 60, currentFar = 1000;

    this.camera = new THREE.PerspectiveCamera(fov, this.width / this.height, 0.1, currentFar);
    this.camera.position.set(2, -3, 0.2);
    this.camera.up.set(0, 0, 1);

    this.setCameraToFitScene = function () {
        var camDistance,
            sceneBoundingBox = startingSceneBoundingBox.clone(),
            objectBoundingBox,
            sceneBoundingSphere;

        if (!K3D.parameters.cameraAutoFit) {
            return;
        }

        this.K3DObjects.traverse(function (object) {
            if (object.geometry && object.geometry.boundingSphere.radius > 0) {
                objectBoundingBox = object.geometry.boundingSphere.getBoundingBox();
                objectBoundingBox.applyMatrix4(object.matrixWorld);
                sceneBoundingBox.union(objectBoundingBox);
            }
        });

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

        sceneBoundingSphere = sceneBoundingBox.getBoundingSphere();

        camDistance = Math.max(sceneBoundingSphere.radius, 0.5) / Math.sin(THREE.Math.degToRad(fov / 2.0));

        this.camera.position.subVectors(
            sceneBoundingSphere.center,
            this.camera.getWorldDirection().setLength(camDistance)
        );
        this.controls.target = sceneBoundingSphere.center;

        if (currentFar * 0.75 < camDistance) {
            this.camera.far = currentFar = camDistance * 1.25;
            this.camera.updateProjectionMatrix();
        }
    };
};
