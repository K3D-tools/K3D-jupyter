'use strict';

var screenfull = require('screenfull');

function fullscreenGUI(container, gui, currentWindow) {
    var obj = {
            fullscreen: false
        },
        controller;

    controller = gui.add(obj, 'fullscreen').name('Full screen').onChange(function (value) {
        if (value) {
            screenfull.request(container);
        } else {
            screenfull.exit();
        }
    });

    currentWindow.addEventListener(screenfull.raw.fullscreenchange, function () {
        obj.fullscreen = screenfull.isFullscreen;

        controller.updateDisplay();
        currentWindow.dispatchEvent(new Event('resize'));
    });
}

module.exports = {
    isAvailable: function () {
        return screenfull.isEnabled;
    },

    initialize: fullscreenGUI
};
