// This file contains the javascript that is run when the notebook is loaded.
// It contains some requirejs configuration and the `load_ipython_extension`
// which is required for any notebook extension.

// Configure requirejs
if (window.require) {
    window.require.config({
        map: {
            '*': {
                'k3d': 'nbextensions/k3d/index',
                'jupyter-widgets-controls': 'nbextensions/jupyter-widgets-controls/extension'
            }
        }
    });
}

// Export the required load_ipython_extention
module.exports = {
    load_ipython_extension: function () {
    }
};
