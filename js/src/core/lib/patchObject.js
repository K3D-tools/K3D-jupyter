'use strict';

var jsonpatch = require('fast-json-patch'),
    error = require('./Error').error,
    removeObject = require('./RemoveObject');

/**
 * @method K3D.getPatchObject
 * @description K3D get objects patch
 * @memberof K3D
 * @param {K3D.Core} K3D A K3D instance to load objects into
 * @param {String} id
 */
function getPatchObject(K3DInstance, id) {

    var object = K3DInstance.Provider.Helpers.getObjectById(K3DInstance.getWorld(), id),
        newObjectJson;

    if (!object) {
        error('Get Patch Object Error', 'K3D get patch object failed, please consult browser error console!', false);
        return;
    }

    if (object.getJson) {
        newObjectJson = object.getJson();
    } else {
        return [];
    }

    return jsonpatch.compare(object.lastSynchJsonObject, newObjectJson);
}

/**
 * @method K3D.applyPatchObject
 * @description K3D apply objects patches
 * @memberof K3D
 * @param {K3D.Core} K3D A K3D instance to load objects into
 * @param {String} id
 * @param {Object} patches
 */
function applyPatchObject(K3DInstance, id, patches) {

    var object = K3DInstance.Provider.Helpers.getObjectById(K3DInstance.getWorld(), id);

    if (!object) {
        error('Apply Patch Object Error',
            'K3D apply patches object failed, please consult browser error console!', false);
        return;
    }

    jsonpatch.apply(object.lastSynchJsonObject, patches);
    removeObject(K3DInstance, id);
    K3DInstance.load({objects: [object.lastSynchJsonObject]});
}

module.exports = {
    getPatchObject: getPatchObject,
    applyPatchObject: applyPatchObject
};
