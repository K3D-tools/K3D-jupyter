const cameraModes = {
    trackball: 'trackball',
    fly: 'fly',
    orbit: 'orbit',
};

function cameraModeGUI(gui, K3D) {
    gui.add(K3D.parameters, 'cameraMode', {
        Trackball: cameraModes.trackball,
        Orbit: cameraModes.orbit,
        Fly: cameraModes.fly,
    }).name('Camera').onChange(
        (mode) => {
            K3D.setCameraMode(mode);

            K3D.dispatch(K3D.events.PARAMETERS_CHANGE, {
                key: 'camera_mode',
                value: mode,
            });
        },
    );
}

module.exports = {
    cameraModeGUI,
    cameraModes,
};
