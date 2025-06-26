const threeMeshBVH = require('three-mesh-bvh');
const StandardInteractions = require('../interactions/StandardCallback');

module.exports = {
    init(config, object, K3D, InteractionsCallback) {
        object.startInteraction = function () {
            if (!object.interactions) {
                object.geometry.boundsTree = new threeMeshBVH.MeshBVH(object.geometry);

                if (InteractionsCallback) {
                    object.interactions = InteractionsCallback(object, K3D);
                } else {
                    object.interactions = StandardInteractions(object, K3D);
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
            if ((changes.click_callback || changes.hover_callback)) {
                obj.startInteraction();
            }

            if (!(changes.click_callback || changes.hover_callback)) {
                obj.stopInteraction();
            }

            resolvedChanges.click_callback = null;
            resolvedChanges.hover_callback = null;
        }
    },
};
