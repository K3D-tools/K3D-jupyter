const LineSimple = require('./LineSimple');
const LineMesh = require('./LineMesh');
const LineThick = require('./LineThick');

/**
 * Loader strategy to handle Line object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config, K3D) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 0xff00;
        config.shader = typeof (config.shader) !== 'undefined' ? config.shader : 'simple';

        if (config.shader === 'mesh') {
            return LineMesh.create(config, K3D);
        }
        if (config.shader === 'simple') {
            return LineSimple.create(config, K3D);
        }
        return LineThick.create(config, K3D);
    },

    update(config, changes, obj, K3D) {
        if (config.shader === 'mesh') {
            return LineMesh.update(config, changes, obj, K3D);
        }
        if (config.shader === 'simple') {
            return LineSimple.update(config, changes, obj, K3D);
        }
        return LineThick.update(config, changes, obj, K3D);
    },
};
