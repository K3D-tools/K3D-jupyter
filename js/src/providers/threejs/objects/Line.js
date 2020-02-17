'use strict';

var LineSimple = require('./LineSimple'),
    LineMesh = require('./LineMesh'),
    LineThick = require('./LineThick');

/**
 * Loader strategy to handle Line object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config, K3D) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 0xff00;
        config.shader = typeof (config.shader) !== 'undefined' ? config.shader : 'simple';

        if (config.shader === 'mesh') {
            return new LineMesh.create(config, K3D);
        } else if (config.shader === 'simple') {
            return new LineSimple.create(config, K3D);
        } else {
            return new LineThick.create(config, K3D);
        }
    },

    update: function (config, changes, obj, K3D) {
        if (config.shader === 'mesh') {
            return LineMesh.update(config, changes, obj, K3D);
        } else if (config.shader === 'simple') {
            return LineSimple.update(config, changes, obj, K3D);
        } else {
            return LineThick.update(config, changes, obj, K3D);
        }
    }
};
