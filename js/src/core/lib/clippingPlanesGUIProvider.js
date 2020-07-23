'use strict';

var planeHelper = require('./helpers/planeGUI');

function clippingPlanesGUIProvider(K3D, clippingPlanesGUI) {

    function dispatch() {
        K3D.dispatch(K3D.events.PARAMETERS_CHANGE, {
            key: 'clipping_planes',
            value: K3D.parameters.clippingPlanes
        });
    }

    planeHelper.init(K3D, clippingPlanesGUI, K3D.parameters.clippingPlanes, 'clippingPlanes', dispatch);
}

module.exports = clippingPlanesGUIProvider;
