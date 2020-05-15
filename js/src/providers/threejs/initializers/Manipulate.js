'use strict';

var THREE = require('three'),
    viewModes = require('./../../../core/lib/viewMode').viewModes;

/**
 * Manipulate service initializer for Three.js library
 * @memberof K3D.Providers.ThreeJS.Initializers
 */
module.exports = function (K3D) {
    var world = K3D.getWorld(), draggingState = false;

    K3D.on(K3D.events.VIEW_MODE_CHANGE, function (mode) {
        if (mode === viewModes.manipulate) {
            world.K3DObjects.children.forEach(function (obj) {
                if (!obj.transformControls && world.ObjectsListJson[obj.K3DIdentifier].model_matrix) {
                    var control = new THREE.TransformControls(world.camera, world.renderer.domElement);

                    control.addEventListener('change', function () {
                        // K3D.dispatch(K3D.events.OBJECT_CHANGE, {
                        //     id: obj.K3DIdentifier,
                        //     key: 'model_matrix',
                        //     value: {
                        //         data: new Float32Array(obj.matrixWorld.elements),
                        //         shape: [4, 4]
                        //     }
                        // });

                        world.render();
                    });

                    control.addEventListener('dragging-changed', function (event) {
                        world.controls.enabled = !event.value;

                        if (draggingState === false && event.value) {
                            draggingState = true;

                            world.K3DObjects.children.forEach(function (obj) {
                                if (obj.transformControls && obj.transformControls !== event.target) {
                                    obj.transformControls.enabled = false;
                                }
                            });
                        }

                        if (draggingState === true && !event.value) {
                            draggingState = false;

                            world.K3DObjects.children.forEach(function (obj) {
                                if (obj.transformControls) {
                                    obj.transformControls.enabled = true;
                                }
                            });
                        }
                    });
                    control.setMode(K3D.parameters.manipulateMode);
                    world.scene.add(control);

                    control.attach(obj);
                    obj.transformControls = control;
                }
            });
        } else {
            world.K3DObjects.children.forEach(function (obj) {
                if (obj.transformControls) {
                    obj.transformControls.detach();
                    obj.transformControls.dispose();
                    delete obj.transformControls;
                }
            });
        }
    });

    K3D.on(K3D.events.MANIPULATE_MODE_CHANGE, function (manipulateMode) {
        world.K3DObjects.children.forEach(function (obj) {
            if (obj.transformControls) {
                if (['translate', 'rotate', 'scale'].indexOf(manipulateMode) !== -1) {
                    obj.transformControls.setMode(manipulateMode);
                }
            }
        });
    });
};
