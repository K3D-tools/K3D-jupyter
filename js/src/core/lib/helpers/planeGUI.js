'use strict';
var THREE = require('three');

function init(K3D, gui, obj, prefix, dispatch) {

    function change(render) {
        obj.length = 0;

        Object.keys(gui[prefix].map).forEach(function (key) {
            if (key !== 'addNew') {
                var data = gui[prefix].map[key].obj;

                obj.push([data.x, data.y, data.z, data.constant]);
            }
        });

        if (render) {
            K3D.render();
        }

        dispatch(obj);
    }

    function refresh(current, render) {
        var camera = K3D.getWorld().camera,
            plane;

        camera.updateMatrixWorld();
        plane = new THREE.Plane().setFromCoplanarPoints(
            new THREE.Vector3(-1, -1, -1).unproject(camera),
            new THREE.Vector3(1, 1, -1).unproject(camera),
            new THREE.Vector3(1, -1, -1).unproject(camera)
        );

        plane.constant -= current.obj.camDist;

        current.obj.x = plane.normal.x;
        current.obj.y = plane.normal.y;
        current.obj.z = plane.normal.z;
        current.obj.constant = plane.constant;

        current.folder.__controllers.forEach(function (controller) {
            controller.updateDisplay();
        });

        change(render);
    }

    if (typeof (gui[prefix]) === 'undefined') {
        gui[prefix] = {};
    }

    if (typeof (gui[prefix].count) === 'undefined') {
        gui[prefix].count = 1;
    }

    if (typeof (gui[prefix].map) === 'undefined') {
        gui[prefix].map = {};

        gui[prefix].map.addNew = gui.add({
            addPlane: function () {
                obj.push([1, 0, 0, 0]);
                init(K3D, gui, obj, prefix, dispatch);
                K3D.render();
                dispatch(obj);
            }
        }, 'addPlane').name('Add new');
    }

    while (obj.length < Object.keys(gui[prefix].map).length - 1) {
        var i = Object.keys(gui[prefix].map).length - 1,
            controllers = gui[prefix].map[i - 1].folder.__controllers;

        controllers[controllers.length - 1].object.delete(true);
    }

    var index = 0;
    Object.keys(gui[prefix].map).sort().forEach(function (key) {
        if (key !== 'addNew') {
            if (parseInt(key) !== index) {
                gui[prefix].map[index] = gui[prefix].map[key];
                delete gui[prefix].map[key];
            }
            index++;
        }
    });

    obj.forEach(function (plane, i) {
        if (typeof (gui[prefix].map[i]) === 'undefined') {
            var current = gui[prefix].map[i] = {
                folder: gui.addFolder('Plane #' + (gui[prefix].count++)),
                obj: {
                    x: plane[0],
                    y: plane[1],
                    z: plane[2],
                    camDist: K3D.getWorld().camera.near * 5000.0,
                    constant: plane[3]
                }
            };

            current.folder.add(current.obj, 'x').step(0.001).onChange(change.bind(this, true));
            current.folder.add(current.obj, 'y').step(0.001).onChange(change.bind(this, true));
            current.folder.add(current.obj, 'z').step(0.001).onChange(change.bind(this, true));
            current.folder.add(current.obj, 'constant').step(0.001).onChange(change.bind(this, true));
            current.folder.add({
                fromCamera: function () {
                    if (current.eventId) {
                        current.folder.__controllers[4].name('From camera [start]');
                        K3D.off(K3D.events.CAMERA_CHANGE, current.eventId);
                        current.eventId = null;
                    } else {
                        current.folder.__controllers[4].name('From camera [stop]');
                        current.eventId = K3D.on(K3D.events.CAMERA_CHANGE, refresh.bind(this, current));
                        refresh(current, true);
                    }
                }
            }, 'fromCamera').name('From camera [start]');
            current.folder.add(current.obj, 'camDist').name('Distance').onChange(function () {
                if (current.eventId) {
                    refresh(current, true);
                }
            });

            current.folder.__controllers[4].domElement.previousSibling.style.width = '100%';

            current.folder.add({
                delete: function (withoutFireChange) {
                    Object.keys(gui[prefix].map).forEach(function (key) {
                        if (current.obj === gui[prefix].map[key].obj) {
                            var folder = gui[prefix].map[key].folder;

                            folder.close();
                            gui.__ul.removeChild(folder.domElement.parentNode);
                            delete gui.__folders[folder.name];
                            gui.onResize();

                            if (current.eventId) {
                                K3D.off(K3D.events.CAMERA_CHANGE, current.eventId);
                                current.eventId = null;
                            }

                            delete gui[prefix].map[key];
                        }
                    });

                    if (!withoutFireChange) {
                        change(true);
                    }
                }
            }, 'delete').name('Delete');
        } else {
            gui[prefix].map[i].obj.x = plane[0];
            gui[prefix].map[i].obj.y = plane[1];
            gui[prefix].map[i].obj.z = plane[2];
            gui[prefix].map[i].obj.constant = plane[3];

            gui[prefix].map[i].folder.__controllers.forEach(function (controller) {
                controller.updateDisplay();
            });
        }
    });
}

module.exports = {
    init: init
};