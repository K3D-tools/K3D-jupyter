/**
 * Validates and setups the defaults for given object assuming it's an object of given K3D type for further drawing
 * @method validateAndPrepareObject
 * @memberof K3D.Helpers
 * @param  {K3D.Core} K3D current K3D instance
 * @param  {Object} object object to check (typical it's an item form objects collection form a K3D JSON)
 * @returns {Object}
 */

module.exports = function (K3D, object) {
    const isObject = typeof (object) === 'object';
    const hasType = isObject && object.hasOwnProperty('type');
    const hasValidType = hasType && K3D.Provider.hasOwnProperty('Objects')
            && K3D.Provider.Objects.hasOwnProperty(object.type);

    if (!isObject) {
        throw new Error('Object definition should be a valid javascript object!');
    } else if (!hasType) {
        throw new Error('Object definition should have a `type` attribute set!');
    } else if (!hasValidType) {
        throw new Error(`Unknown object type (${object.type}) passed, no loader supporting it found!`);
    }

    return object;
};
