const {viewModes} = require('./viewMode');

const manipulateModes = {
    translate: 'translate',
    scale: 'scale',
    rotate: 'rotate',
};

module.exports = {
    refreshManipulateGUI(K3D, GUI) {
        GUI.controls.controllers.forEach((controller) => {
            if (controller.property === 'manipulateMode') {
                controller.domElement.hidden = K3D.parameters.viewMode !== viewModes.manipulate;
                controller.updateDisplay();
            }
        });
    },
    manipulateGUI(gui, K3D, changeParameters) {
        gui.add(K3D.parameters, 'manipulateMode', {
            Translate: manipulateModes.translate,
            Rotate: manipulateModes.rotate,
            Scale: manipulateModes.scale,
        }).name('Manipulate mode').onChange(
            (mode) => {
                K3D.setManipulateMode(mode);
                changeParameters('manipulate_mode', mode);
            },
        );
    },
    manipulateModes,
};
