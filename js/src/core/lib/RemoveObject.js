'use strict';

var error = require('./Error').error;

/**
 * @method K3D.removeObject
 * @description K3D objects remover
 * @memberof K3D
 * @param {K3D.Core} K3D A K3D instance to load objects into
 * @param {String} id
 */
module.exports = function (K3D, id) {

    try {
        K3D.removeObject(id);

        K3D.rebuild();
        K3D.getWorld().setCameraToFitScene();
        K3D.getWorld().render();
    } catch (e) {
        error('Remove Object Error', 'K3D remove object failed, please consult browser error console!', false);
        throw e;
    }
};
