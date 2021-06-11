const MeshStandard = require('./MeshStandard');
const MeshVolume = require('./MeshVolume');

/**
 * Loader strategy to handle Mesh object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */

function isMeshVolume(config) {
    return config.volume && config.volume.data && config.volume.data.length > 0
        && config.volume_bounds && config.volume_bounds.data && config.volume_bounds.data.length > 0
        && config.color_range && config.color_range.length > 0
        && config.color_map && config.color_map.data && config.color_map.data.length > 0;
}

module.exports = {
    create(config, K3D) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;

        if (isMeshVolume(config)) {
            return MeshVolume.create(config, K3D);
        }
        return MeshStandard.create(config, K3D);
    },

    update(config, changes, obj, K3D) {
        if (isMeshVolume(config)) {
            return MeshVolume.update(config, changes, obj, K3D);
        }
        return MeshStandard.update(config, changes, obj, K3D);
    },
};
