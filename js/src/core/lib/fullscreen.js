const screenfull = require('screenfull').default;

function fullscreenGUI(container, gui, currentWindow, K3D) {
    const obj = {
        fullscreen: false,
    };

    const controller = gui.add(obj, 'fullscreen').name('Full screen').onChange((value) => {
        K3D.heavyOperationSync = true;
        if (value) {
            screenfull.request(container);
        } else {
            screenfull.exit();
        }
    });

    const onFullscreenChange = () => {
        obj.fullscreen = screenfull.isFullscreen;

        controller.updateDisplay();
        currentWindow.dispatchEvent(new Event('resize'));
    };

    currentWindow.addEventListener(screenfull.raw.fullscreenchange, onFullscreenChange);

    // The listener lives on the main window and captures the controller (and through it the
    // K3D instance), while initializeGUI runs again every time the menu is re-shown. Hand the
    // remover back so the GUI teardown can drop it instead of leaking one per init.
    return function removeFullscreenListener() {
        currentWindow.removeEventListener(screenfull.raw.fullscreenchange, onFullscreenChange);
    };
}

module.exports = {
    isAvailable() {
        return screenfull.isEnabled;
    },

    initialize: fullscreenGUI,
    screenfull,
};
