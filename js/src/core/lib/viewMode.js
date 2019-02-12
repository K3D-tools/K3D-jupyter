'use strict';

var viewModes = {
    view: 'view',
    add: 'add',
    change: 'change',
    callback: 'callback'
};

function viewModeGUI(gui, K3D) {
    gui.add(K3D.parameters, 'viewMode', {
        View: viewModes.view,
        Add: viewModes.add,
        Change: viewModes.change,
        Callback: viewModes.callback
    }).name('Mode').onChange(
        function (mode) {
            K3D.setViewMode(mode);
        });
}

module.exports = {
    viewModeGUI: viewModeGUI,
    viewModes: viewModes
};
