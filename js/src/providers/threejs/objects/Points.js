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
    config.visible = typeof(config.visible) !== 'undefined' ? config.visible : true;
    config.color = typeof(config.color) !== 'undefined' ? config.color : 0xff00;
    config.point_size = typeof(config.point_size) !== 'undefined' ? config.point_size : 1.0;
    config.shader = typeof(config.shader) !== 'undefined' ? config.shader : '3dSpecular';

    if (config.shader === 'mesh') {
        return new PointsMesh(config);
    } else {
        return new PointsBillboard(config);
    }
};
