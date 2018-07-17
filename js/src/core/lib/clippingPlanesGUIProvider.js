'use strict';

function clippingPlanesGUIProvider(K3D, clippingPlanesGUI) {

    function dispatch() {
        K3D.dispatch(K3D.events.PARAMETERS_CHANGE, {
            key: 'clipping_planes',
            value: K3D.parameters.clippingPlanes
        });
    }

    function change(render) {
        var clippingPlanes = [];

        Object.keys(K3D.clippingPlanes_map).forEach(function (key) {
            if (key !== 'addNew') {
                var data = K3D.clippingPlanes_map[key].obj;

                clippingPlanes.push([data.x, data.y, data.z, data.constant]);
            }
        });

        K3D.parameters.clippingPlanes = clippingPlanes;

        if (render) {
            K3D.render();
        }

        dispatch();
    }

    if (typeof(K3D.clippingPlanes_count) === 'undefined') {
        K3D.clippingPlanes_count = 1;
    }

    if (typeof(K3D.clippingPlanes_map) === 'undefined') {
        K3D.clippingPlanes_map = {};

        K3D.clippingPlanes_map.addNew = clippingPlanesGUI.add({
            addClippingPlane: function () {
                K3D.parameters.clippingPlanes.push([1, 0, 0, 0]);
                clippingPlanesGUIProvider(K3D, clippingPlanesGUI);
                K3D.render();
                dispatch();
            }
        }, 'addClippingPlane').name('Add new');
    }

    while (K3D.parameters.clippingPlanes.length < Object.keys(K3D.clippingPlanes_map).length - 1) {
        var i = Object.keys(K3D.clippingPlanes_map).length - 1,
            controllers = K3D.clippingPlanes_map[i - 1].folder.__controllers;

        controllers[controllers.length - 1].object.delete(true);
    }

    var index = 0;
    Object.keys(K3D.clippingPlanes_map).sort().forEach(function (key) {
        if (key !== 'addNew') {
            if (parseInt(key) !== index) {
                K3D.clippingPlanes_map[index] = K3D.clippingPlanes_map[key];
                delete K3D.clippingPlanes_map[key];
            }
            index++;
        }
    });

    K3D.parameters.clippingPlanes.forEach(function (plane, i) {
        if (typeof(K3D.clippingPlanes_map[i]) === 'undefined') {
            var current = K3D.clippingPlanes_map[i] = {
                folder: clippingPlanesGUI.addFolder('Clipping Plane #' + (K3D.clippingPlanes_count++)),
                obj: {
                    x: plane[0],
                    y: plane[1],
                    z: plane[2],
                    constant: plane[3]
                }
            };

            current.folder.add(current.obj, 'x').step(0.001).onChange(change.bind(this, true));
            current.folder.add(current.obj, 'y').step(0.001).onChange(change.bind(this, true));
            current.folder.add(current.obj, 'z').step(0.001).onChange(change.bind(this, true));
            current.folder.add(current.obj, 'constant').step(0.001).onChange(change.bind(this, true));
            current.folder.add({
                fromCamera: function () {
                    function refresh() {
                        var camera = K3D.getWorld().camera,
                            plane = new THREE.Plane().setFromCoplanarPoints(
                                new THREE.Vector3(-1, -1, -1).unproject(camera),
                                new THREE.Vector3(1, 1, -1).unproject(camera),
                                new THREE.Vector3(1, -1, -1).unproject(camera)
                            );

                        plane.constant -= camera.near * 200.0;

                        current.obj.x = plane.normal.x;
                        current.obj.y = plane.normal.y;
                        current.obj.z = plane.normal.z;
                        current.obj.constant = plane.constant;

                        current.folder.__controllers.forEach(function (controller) {
                            controller.updateDisplay();
                        });

                        change();
                    }

                    if (current.eventId) {
                        current.folder.__controllers[4].name('From camera [start]');
                        K3D.off(K3D.events.CAMERA_CHANGE, current.eventId);
                        current.eventId = null;
                    } else {
                        current.folder.__controllers[4].name('From camera [stop]');
                        current.eventId = K3D.on(K3D.events.CAMERA_CHANGE, refresh);
                        refresh();
                    }
                }
            }, 'fromCamera').name('From camera [start]');

            current.folder.__controllers[4].domElement.previousSibling.style.width = '100%';

            current.folder.add({
                delete: function (withoutFireChange) {
                    Object.keys(K3D.clippingPlanes_map).forEach(function (key) {
                        if (current.obj === K3D.clippingPlanes_map[key].obj) {
                            var folder = K3D.clippingPlanes_map[key].folder;

                            folder.close();
                            clippingPlanesGUI.__ul.removeChild(folder.domElement.parentNode);
                            delete clippingPlanesGUI.__folders[folder.name];
                            clippingPlanesGUI.onResize();

                            if (current.eventId) {
                                K3D.off(K3D.events.CAMERA_CHANGE, current.eventId);
                                current.eventId = null;
                            }

                            delete K3D.clippingPlanes_map[key];
                        }
                    });

                    if (!withoutFireChange) {
                        change();
                    }
                }
            }, 'delete').name('Delete');
        } else {
            K3D.clippingPlanes_map[i].obj.x = plane[0];
            K3D.clippingPlanes_map[i].obj.y = plane[1];
            K3D.clippingPlanes_map[i].obj.z = plane[2];
            K3D.clippingPlanes_map[i].obj.constant = plane[3];

            K3D.clippingPlanes_map[i].folder.__controllers.forEach(function (controller) {
                controller.updateDisplay();
            });
        }
    });
}

module.exports = clippingPlanesGUIProvider;
