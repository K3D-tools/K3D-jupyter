const cameraUpAxisModes = {
    x: 'x',
    y: 'y',
    z: 'z',
    none: 'none'
};

function cameraUpAxisGUI(gui, K3D) {
    gui.add(K3D.parameters, 'cameraUpAxis', {
        x: cameraUpAxisModes.x,
        y: cameraUpAxisModes.y,
        z: cameraUpAxisModes.z,
        none: cameraUpAxisModes.none
    }).name('CameraUpAxis').onChange(
        (axis) => {
            K3D.setCameraUpAxis(axis);

            K3D.dispatch(K3D.events.PARAMETERS_CHANGE, {
                key: 'camera_up_axis',
                value: axis
            });
        },
    );
}

function setupUpVector(camera, cameraUpAxis) {
    if (cameraUpAxis === cameraUpAxisModes.x) {
        camera.up.set(1, 0, 0);
    } else if (cameraUpAxis === cameraUpAxisModes.y) {
        camera.up.set(0, 1, 0);
    } else if (cameraUpAxis === cameraUpAxisModes.z) {
        camera.up.set(0, 0, 1);
    }
}

module.exports = {
    cameraUpAxisGUI,
    cameraUpAxisModes,
    setupUpVector
};
