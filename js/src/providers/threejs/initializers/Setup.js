/**
 * Setup what's required before initializing the rest
 * @this K3D.Core~world
 * @method Setup
 * @memberof K3D.Providers.ThreeJS.Initializers
 */

const { getSpaceDimensionsFromTargetElement } = require('../helpers/Fn');

module.exports = function () {
    const dimensions = getSpaceDimensionsFromTargetElement(this);

    this.width = dimensions[0];
    this.height = dimensions[1];

    this.axesHelper = {
        width: 100,
        height: 100,
    };
};
