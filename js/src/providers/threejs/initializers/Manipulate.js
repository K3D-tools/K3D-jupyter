const THREE = require('three');
const { viewModes } = require('../../../core/lib/viewMode');

/**
 * Manipulate service initializer for Three.js library
 * @memberof K3D.Providers.ThreeJS.Initializers
 */
module.exports = function (K3D) {
    const world = K3D.getWorld(); let
        draggingState = false;

    K3D.on(K3D.events.VIEW_MODE_CHANGE, (mode) => {
        if (mode === viewModes.manipulate) {
            world.K3DObjects.children.forEach((obj) => {
                if (!obj.transformControls && world.ObjectsListJson[obj.K3DIdentifier].model_matrix) {
                    const control = new THREE.TransformControls(world.camera, world.renderer.domElement);

                    control.addEventListener('change', () => {
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

                    control.addEventListener('dragging-changed', (event) => {
                        world.controls.enabled = !event.value;

                        if (draggingState === false && event.value) {
                            draggingState = true;

                            world.K3DObjects.children.forEach((o) => {
                                if (o.transformControls && o.transformControls !== event.target) {
                                    o.transformControls.enabled = false;
                                }
                            });
                        }

                        if (draggingState === true && !event.value) {
                            draggingState = false;

                            world.K3DObjects.children.forEach((o) => {
                                if (o.transformControls) {
                                    o.transformControls.enabled = true;
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
            world.K3DObjects.children.forEach((obj) => {
                if (obj.transformControls) {
                    obj.transformControls.detach();
                    obj.transformControls.dispose();
                    delete obj.transformControls;
                }
            });
        }
    });

    K3D.on(K3D.events.MANIPULATE_MODE_CHANGE, (manipulateMode) => {
        world.K3DObjects.children.forEach((obj) => {
            if (obj.transformControls) {
                if (['translate', 'rotate', 'scale'].indexOf(manipulateMode) !== -1) {
                    obj.transformControls.setMode(manipulateMode);
                }
            }
        });
    });
};
