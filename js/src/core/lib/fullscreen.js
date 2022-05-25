const screenfull = require('screenfull').default;

function fullscreenGUI(container, gui, currentWindow) {
    const obj = {
        fullscreen: false,
    };

    const controller = gui.add(obj, 'fullscreen').name('Full screen').onChange((value) => {
        if (value) {
            screenfull.request(container);
        } else {
            screenfull.exit();
        }
    });

    currentWindow.addEventListener(screenfull.raw.fullscreenchange, () => {
        obj.fullscreen = screenfull.isFullscreen;

        controller.updateDisplay();
        currentWindow.dispatchEvent(new Event('resize'));
    });
}

module.exports = {
    isAvailable() {
        return screenfull.isEnabled;
    },

    initialize: fullscreenGUI,
    screenfull: screenfull
};
