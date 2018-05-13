'use strict';

var LineSimple = require('./LineSimple'),
    LineMesh = require('./LineMesh');

/**
 * Loader strategy to handle Line object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {
    var shader = config.shader || 'simple';

    if (shader === 'mesh') {
        return new LineMesh(config);
    } else {
        return new LineSimple(config);
    }
};
