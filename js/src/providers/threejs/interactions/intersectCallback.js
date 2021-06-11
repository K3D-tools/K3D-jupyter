const { viewModes } = require('../../../core/lib/viewMode');

/**
 * Interactions handlers for Voxels object
 * @method Voxels
 * @memberof K3D.Providers.ThreeJS.Interactions
 */
module.exports = function (object, K3D) {
    function onClickCallback(intersect) {
        K3D.dispatch(K3D.events.OBJECT_CLICKED, intersect);

        return false;
    }

    function onHoverCallback(intersect) {
        K3D.dispatch(K3D.events.OBJECT_HOVERED, intersect);

        return false;
    }

    return {
        onHover(intersect, viewMode) {
            switch (viewMode) {
                case viewModes.callback:
                    return onHoverCallback(intersect);
                default:
                    return false;
            }
        },
        onClick(intersect, viewMode) {
            switch (viewMode) {
                case viewModes.callback:
                    return onClickCallback(intersect);
                default:
                    return false;
            }
        },
    };
};
