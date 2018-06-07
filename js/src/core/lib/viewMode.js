'use strict';

var viewModes = {
    view: 'view',
    add: 'add',
    change: 'change',
    callback: 'callback'
};

function viewModeGUI(gui, K3D) {

    var obj = {
        mode: K3D.parameters.viewMode
    };

    gui.add(obj, 'mode', {
        View: viewModes.view,
        Add: viewModes.add,
        Change: viewModes.change,
        Callback: viewModes.callback
    }).name('Mode').onChange(
        function (mode) {
            K3D.setViewMode(mode);
        });

    K3D.setViewMode(obj.mode);
}

module.exports = {
    viewModeGUI: viewModeGUI,
    viewModes: viewModes
};
