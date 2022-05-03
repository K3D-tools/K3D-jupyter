const LinesSimple = require('./LinesSimple');
const LinesMesh = require('./LinesMesh');
const LinesThick = require('./LinesThick');

/**
 * Loader strategy to handle Lines object
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
            return LinesMesh.create(config, K3D);
        } if (config.shader === 'simple') {
            return LinesSimple.create(config, K3D);
        }
        return LinesThick.create(config, K3D);
    },

    update(config, changes, obj, K3D) {
        if (config.shader === 'mesh') {
            return LinesMesh.update(config, changes, obj, K3D);
        } if (config.shader === 'simple') {
            return LinesSimple.update(config, changes, obj, K3D);
        }
        return LinesThick.update(config, changes, obj, K3D);
    },
};
