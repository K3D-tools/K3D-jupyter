const THREE = require('three');
const threeMeshBVH = require('three-mesh-bvh');

require('./helpers/THREE.STLLoader')(THREE);
require('./helpers/THREE.CopyShader')(THREE);
require('./helpers/THREE.TrackballControls')(THREE);
require('./helpers/THREE.SliceControls')(THREE);
require('./helpers/THREE.VolumeSidesControls')(THREE);
require('./helpers/THREE.OrbitControls')(THREE);

THREE.TransformControls = require('./helpers/TransformControls').TransformControls;

THREE.Mesh.prototype.raycast = threeMeshBVH.acceleratedRaycast;

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
        Setup: require('./initializers/Setup'),
        Manipulate: require('./initializers/Manipulate'),
    },
    /**
     * @namespace Objects
     * @memberof! K3D.Providers.ThreeJS
     * @type {Object}
     */
    Objects: {
        Line: require('./objects/Line'),
        Lines: require('./objects/Lines'),
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
        Volume: require('./objects/Volume'),
        MIP: require('./objects/MIP'),
        Label: require('./objects/Label'),
        VolumeSlice: require('./objects/VolumeSlice'),
    },
    /**
     * @namespace Interactions
     * @memberof! K3D.Providers.ThreeJS
     * @type {Object}
     */
    Interactions: {
        Voxels: require('./interactions/Voxels'),
        PointsCallback: require('./interactions/PointsCallback'),
        StandardCallback: require('./interactions/StandardCallback'),
    },

    THREE,
};
