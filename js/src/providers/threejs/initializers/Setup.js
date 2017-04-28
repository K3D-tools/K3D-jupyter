'use strict';

/**
 * Setup what's required before initializing the rest
 * @this K3D.Core~world
 * @method Setup
 * @memberof K3D.Providers.ThreeJS.Initializers
 */

var getSpaceDimensionsFromTargetElement = require('./../helpers/Fn').getSpaceDimensionsFromTargetElement;

module.exports = function () {

    var dimensions = getSpaceDimensionsFromTargetElement(this);

    this.width = dimensions[0];
    this.height = dimensions[1];
};
