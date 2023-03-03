/**
 * Interactions handlers for standard object
 * @memberof K3D.Providers.ThreeJS.Interactions
 */

function prepareParam(param) {
    return {
        position: param.point.toArray(),
        distance: param.distance,
        index: param.index,
        K3DIdentifier: param.object.K3DIdentifier,
    };
}

module.exports = function (object, K3D) {
    return {
        onHover(intersect) {
            K3D.dispatch(K3D.events.OBJECT_HOVERED, prepareParam(intersect));
            return false;
        },
        onClick(intersect) {
            K3D.dispatch(K3D.events.OBJECT_CLICKED, prepareParam(intersect));
            return false;
        },
    };
};
