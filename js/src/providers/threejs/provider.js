'use strict';

var THREE = require('three');

require('./helpers/THREE.Octree')(THREE);
require('./helpers/THREE.STLLoader')(THREE);
require('./helpers/THREE.CopyShader')(THREE);
require('./helpers/THREE.TrackballControls')(THREE);

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
        SparseVoxels: require('./objects/SparseVoxels'),
        VoxelsGroup: require('./objects/VoxelsGroup'),
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
