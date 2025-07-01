const THREE = require('three');

function init(K3D, gui, obj, prefix, dispatch) {
    function change(render) {
        obj.length = 0;

        Object.keys(gui[prefix].map).forEach((key) => {
            if (key !== 'addNew') {
                const data = gui[prefix].map[key].obj;

                obj.push([data.x, data.y, data.z, data.constant]);
            }
        });

        if (render) {
            K3D.render();
        }

        dispatch(obj);
    }

    function refresh(current, render) {
        const { camera } = K3D.getWorld();
        camera.updateMatrixWorld();
        const plane = new THREE.Plane().setFromCoplanarPoints(
            new THREE.Vector3(-1, -1, -1).unproject(camera),
            new THREE.Vector3(1, 1, -1).unproject(camera),
            new THREE.Vector3(1, -1, -1).unproject(camera),
        );

        plane.constant -= current.obj.camDist;

        current.obj.x = plane.normal.x;
        current.obj.y = plane.normal.y;
        current.obj.z = plane.normal.z;
        current.obj.constant = plane.constant;

        current.folder.controllers.forEach((controller) => {
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
            addPlane() {
                obj.push([1, 0, 0, 0]);
                init(K3D, gui, obj, prefix, dispatch);
                K3D.render();
                dispatch(obj);
            },
        }, 'addPlane').name('Add new');
    }

    while (obj.length < Object.keys(gui[prefix].map).length - 1) {
        const i = Object.keys(gui[prefix].map).length - 1;
        const controllers = gui[prefix].map[i - 1].folder.controllers;

        controllers[controllers.length - 1].object.delete(true);
    }

    let index = 0;
    Object.keys(gui[prefix].map).sort().forEach((key) => {
        if (key !== 'addNew') {
            if (parseInt(key, 10) !== index) {
                gui[prefix].map[index] = gui[prefix].map[key];
                delete gui[prefix].map[key];
            }
            index += 1;
        }
    });

    obj.forEach(function (plane, i) {
        if (typeof (gui[prefix].map[i]) === 'undefined') {
            const current = {
                folder: gui.addFolder(`Plane #${gui[prefix].count++}`),
                obj: {
                    x: plane[0],
                    y: plane[1],
                    z: plane[2],
                    camDist: K3D.getWorld().camera.near * 5000.0,
                    constant: plane[3],
                },
            };

            gui[prefix].map[i] = current;

            current.folder.add(current.obj, 'x').step(0.001).onChange(change.bind(this, true));
            current.folder.add(current.obj, 'y').step(0.001).onChange(change.bind(this, true));
            current.folder.add(current.obj, 'z').step(0.001).onChange(change.bind(this, true));
            current.folder.add(current.obj, 'constant').step(0.001).onChange(change.bind(this, true));
            current.folder.add({
                fromCamera() {
                    if (current.eventId) {
                        current.folder.controllers[4].name('From camera [start]');
                        K3D.off(K3D.events.CAMERA_CHANGE, current.eventId);
                        current.eventId = null;
                    } else {
                        current.folder.controllers[4].name('From camera [stop]');
                        current.eventId = K3D.on(K3D.events.CAMERA_CHANGE, refresh.bind(this, current));
                        refresh(current, true);
                    }
                },
            }, 'fromCamera').name('From camera [start]');
            current.folder.add(current.obj, 'camDist').name('Distance').onChange(() => {
                if (current.eventId) {
                    refresh(current, true);
                }
            });

            current.folder.controllers[4].domElement.previousSibling.style.width = '100%';

            current.folder.add({
                delete(withoutFireChange) {
                    Object.keys(gui[prefix].map).forEach((key) => {
                        if (current.obj === gui[prefix].map[key].obj) {
                            const { folder } = gui[prefix].map[key];

                            folder.destroy();

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
                },
            }, 'delete').name('Delete');
        } else {
            gui[prefix].map[i].obj.x = plane[0];
            gui[prefix].map[i].obj.y = plane[1];
            gui[prefix].map[i].obj.z = plane[2];
            gui[prefix].map[i].obj.constant = plane[3];

            gui[prefix].map[i].folder.controllers.forEach((controller) => {
                controller.updateDisplay();
            });
        }
    });
}

module.exports = {
    init,
};
