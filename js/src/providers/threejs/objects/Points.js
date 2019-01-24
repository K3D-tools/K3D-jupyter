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
module.exports = {
    create: function (config) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 0xff00;
        config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;
        config.point_size = typeof (config.point_size) !== 'undefined' ? config.point_size : 1.0;
        config.shader = typeof (config.shader) !== 'undefined' ? config.shader : '3dSpecular';

        config.opacity = Math.max(Math.min(config.opacity, 1.0), 0.0);

        if (config.shader === 'mesh') {
            return new PointsMesh.create(config);
        } else {
            return new PointsBillboard.create(config);
        }
    }
};
