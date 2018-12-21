'use strict';

var TextureImage = require('./TextureImage'),
    TextureData = require('./TextureData');

/**
 * Loader strategy to handle Texture object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config, K3D) {
        config.visible = typeof(config.visible) !== 'undefined' ? config.visible : true;

        if (config.file_format && config.binary) {
            return new TextureImage.create(config, K3D);
        } else {
            return new TextureData.create(config, K3D);
        }
    }
};
