const threeMeshBVH = require('three-mesh-bvh');
const StandardInteractions = require('../interactions/StandardCallback');

module.exports = {
    init(config, object, K3D, InteractionsCallback, geometry, Intersect) {
        object.startInteraction = function () {
            if (!object.interactions) {
                if (typeof (geometry) === 'undefined') {
                    geometry = object.geometry;
                }

                // indirect keeps the index buffer as it came, so faceIndex stays the triangle
                // number the user sent; the points path maps the slot itself and needs the sort
                object.geometry.boundsTree = new threeMeshBVH.MeshBVH(geometry, {
                    indirect: typeof (Intersect) === 'undefined',
                });

                if (InteractionsCallback) {
                    object.interactions = InteractionsCallback(object, K3D);
                } else {
                    object.interactions = StandardInteractions(object, K3D);
                }

                if (typeof (Intersect) !== 'undefined') {
                    object.interactions.intersect = Intersect(object, K3D);
                }
            }
        };

        object.stopInteraction = function () {
            if (object.interactions) {
                object.geometry.boundsTree = null;
                object.interactions = null;
            }
        };

        if (config.click_callback || config.hover_callback) {
            object.startInteraction();
        }
    },

    update(config, changes, resolvedChanges, obj) {
        if (typeof (changes.click_callback) !== 'undefined' || typeof (changes.hover_callback) !== 'undefined') {
            // config carries both flags as they stand; changes names only the one that moved
            if (config.click_callback || config.hover_callback) {
                obj.startInteraction();
            } else {
                obj.stopInteraction();
            }

            resolvedChanges.click_callback = null;
            resolvedChanges.hover_callback = null;
        }
    },
};
