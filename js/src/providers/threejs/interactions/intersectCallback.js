
/**
 * Interactions handlers for Voxels object
 * @method Voxels
 * @memberof K3D.Providers.ThreeJS.Interactions
 */
module.exports = function (object, K3D) {
    return {
        onHover(intersect) {
            K3D.dispatch(K3D.events.OBJECT_HOVERED, intersect);
            return false;
        },
        onClick(intersect) {
            K3D.dispatch(K3D.events.OBJECT_CLICKED, intersect);
            return false;
        },
    };
};
