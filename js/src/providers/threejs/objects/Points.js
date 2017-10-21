'use strict';

var PointsMesh = require('./PointsMesh'),
    PointsBillboard = require('./PointsBillboard');

/**
 * Loader strategy to handle Points object
 * @method Points
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {
    var shader = config.shader || '3dSpecular';

    if (shader === 'mesh') {
        return new PointsMesh(config);
    } else {
        return new PointsBillboard(config);
    }
};
