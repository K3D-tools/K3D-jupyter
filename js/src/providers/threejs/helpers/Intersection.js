'use strict';

var threeMeshBVH = require('three-mesh-bvh'),
    intersectCallback = require('./../interactions/intersectCallback');
// THREE = require('three'),
// viewModes = require('./../../../core/lib/viewMode').viewModes;

module.exports = {
    init: function (config, object, K3D) {

        object.startInteraction = function () {
            if (!object.interactions) {
                object.geometry.boundsTree = new threeMeshBVH.MeshBVH(object.geometry);
                object.interactions = intersectCallback(object, K3D);
            }
        };

        object.stopInteraction = function () {
            if (object.interactions) {
                object.geometry.boundsTree = null;
                object.interactions = null;
            }
        };

        // if (config.click_callback || config.hover_callback || K3D.parameters.viewMode === viewModes.manipulate) {
        if (config.click_callback || config.hover_callback) {
            object.startInteraction();
        }

        // object.startInteraction();
        //
        // var o = new THREE.Group(),
        //     helper = new threeMeshBVH.Visualizer(object);
        // helper.depth = 7;
        // helper.update();
        //
        // o.add(object);
        // o.add(helper);
        //
        // return o;
    },

    update: function (config, changes, resolvedChanges, obj) {
        if (typeof(changes.click_callback) !== 'undefined' || typeof(changes.hover_callback) !== 'undefined') {
            if ((changes.click_callback || changes.hover_callback)) {
                obj.startInteraction();
            }

            if (!(changes.click_callback || changes.hover_callback)) {
                obj.stopInteraction();
            }

            resolvedChanges.click_callback = null;
            resolvedChanges.hover_callback = null;
        }
    }
};