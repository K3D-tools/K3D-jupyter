'use strict';

window.THREE = require('three');
require('./../../../node_modules/three/examples/js/Detector');
require('./../../../node_modules/three/examples/js/Octree');
require('./../../../node_modules/three/examples/js/loaders/STLLoader');
require('./../../../node_modules/three/examples/js/shaders/CopyShader');
require('./../../../node_modules/three/examples/js/postprocessing/EffectComposer');
require('./../../../node_modules/three/examples/js/postprocessing/SSAARenderPass');

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
        Scene: require('./initializers/Scene').Init,
        Setup: require('./initializers/Setup')
    },
    /**
     * @namespace Objects
     * @memberof! K3D.Providers.ThreeJS
     * @type {Object}
     */
    Objects: {
        Line: require('./objects/Line'),
        MarchingCubes: require('./objects/MarchingCubes'),
        Mesh: require('./objects/Mesh'),
        Points: require('./objects/Points'),
        STL: require('./objects/STL'),
        Surface: require('./objects/Surface'),
        TextureText: require('./objects/TextureText'),
        Texture: require('./objects/Texture'),
        Text: require('./objects/Text'),
        Text2d: require('./objects/Text2d'),
        TorusKnot: require('./objects/TorusKnot'),
        VectorField: require('./objects/VectorField'),
        Vectors: require('./objects/Vectors'),
        Voxels: require('./objects/Voxels'),
        Volume: require('./objects/Volume')
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
