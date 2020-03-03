'use strict';

var viewModes = require('./viewMode').viewModes,
    manipulateModes = {
        translate: 'translate',
        scale: 'scale',
        rotate: 'rotate'
    };

module.exports = {
    refreshManipulateGUI: function (K3D, GUI) {
        GUI.controls.__controllers.forEach(function (controller) {
            if (controller.property === 'manipulateMode') {
                controller.__li.hidden = K3D.parameters.viewMode !== viewModes.manipulate;
                controller.updateDisplay();
            }
        });
    },
    manipulateGUI: function (gui, K3D, changeParameters) {
        gui.add(K3D.parameters, 'manipulateMode', {
            Translate: manipulateModes.translate,
            Rotate: manipulateModes.rotate,
            Scale: manipulateModes.scale
        }).name('Manipulate mode').onChange(
            function (mode) {
                K3D.setManipulateMode(mode);
                changeParameters('manipulate_mode', mode);
            });
    },
    manipulateModes: manipulateModes
};
