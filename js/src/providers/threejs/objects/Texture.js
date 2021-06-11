const TextureImage = require('./TextureImage');
const TextureData = require('./TextureData');

/**
 * Loader strategy to handle Texture object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config, K3D) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;

        if (config.file_format && config.binary) {
            return TextureImage.create(config, K3D);
        }
        return TextureData.create(config, K3D);
    },

    update(config, changes, obj, K3D) {
        if (config.file_format && config.binary) {
            return TextureImage.update(config, changes, obj, K3D);
        }
        return TextureData.update(config, changes, obj, K3D);
    },
};
