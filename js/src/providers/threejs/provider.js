'use strict';

window.THREE = require('three');
require('./../../../node_modules/three/examples/js/Detector');
require('./../../../node_modules/three/examples/js/Octree');
require('./../../../node_modules/three/examples/js/loaders/STLLoader');

require('./helpers/TrackballControls');


/**
 * K3D ThreeJS Provider namespace
 * @alias module:ThreeJS
 * @namespace ThreeJS
 * @memberof K3D.Providers
 * @property {object} Helpers          - sub-namespace for registering Helpers
 * @property {object} Initializers     - sub-namespace for all Initializers
 * @property {object} Objects          - sub-namespace for all objects
 */

module.exports = {
    /**
     * @namespace Helpers
     * @memberof! K3D.Providers.ThreeJS
     * @type {Object}
     */
    Helpers: require('./helpers/Fn'),
    /**
     * @namespace Initializers
     * @memberof! K3D.Providers.ThreeJS
     * @type {Object}
     */
    Initializers: {
        Canvas: require('./initializers/Canvas'),
        Camera: require('./initializers/Camera'),
        Renderer: require('./initializers/Renderer'),
        Scene: require('./initializers/Scene'),
        Setup: require('./initializers/Setup')
    },
    /**
     * @namespace Objects
     * @memberof! K3D.Providers.ThreeJS
     * @type {Object}
     */
    Objects: {
        Text2d: require('./objects/Text2d'),
        Line: require('./objects/Line'),
        MarchingCubes: require('./objects/MarchingCubes'),
        Points: require('./objects/Points'),
        STL: require('./objects/STL'),
        Surface: require('./objects/Surface'),
        Text: require('./objects/Text'),
        Texture: require('./objects/Texture'),
        TorusKnot: require('./objects/TorusKnot'),
        Vectors: require('./objects/Vectors'),
        VectorsFields: require('./objects/VectorsFields'),
        Voxels: require('./objects/Voxels'),
        Mesh: require('./objects/Mesh')
    },
    /**
     * @namespace Interactions
     * @memberof! K3D.Providers.ThreeJS
     * @type {Object}
     */
    Interactions: {
        Voxels: require('./interactions/Voxels')
    }
};
