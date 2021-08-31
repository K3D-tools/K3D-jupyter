// This file contains the javascript that is run when the notebook is loaded.
// It contains some requirejs configuration and the `load_ipython_extension`
// which is required for any notebook extension.

// Configure requirejs
if (window.require) {
    window.require.config({
        map: {
            '*': {
                k3d: 'nbextensions/k3d/index',
                'jupyter-widgets-controls': 'nbextensions/jupyter-widgets-controls/extension',
            },
        },
    });
}

window.__webpack_public_path__ = `${document.querySelector('body').getAttribute('data-base-url')}nbextensions/k3d/`;

// require('style-loader?{attributes:{id: "k3d-katex"}}!css-loader!katex/dist/katex.min.css');
// require('style-loader?{attributes:{id: "k3d-dat.gui"}}!css-loader!dat.gui/build/dat.gui.css');

require('katex/dist/katex.min.css');
require('dat.gui/build/dat.gui.css');

// Export the required load_ipython_extention
module.exports = {
    load_ipython_extension() {
    },
};
