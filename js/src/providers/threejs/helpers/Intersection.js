'use strict';

var threeMeshBVH = require('three-mesh-bvh'),
    intersectCallback = require('./../interactions/intersectCallback');

module.exports = {
    init: function (config, object, K3D) {
        if (config.click_callback || config.hover_callback) {
            object.geometry.boundsTree = new threeMeshBVH.MeshBVH(object.geometry);
            object.interactions = intersectCallback(object, K3D);
        }

        // var o = new THREE.Group(),
        //     helper = new threeMeshBVH.Visualizer(object);
        // helper.depth = 40;
        // helper.update();
        // o.add(object);
        // o.add(helper);
        // return Promise.resolve(o);
    },

    update: function (config, changes, obj, K3D) {
        if (typeof(changes.click_callback) !== 'undefined' || typeof(changes.hover_callback) !== 'undefined') {
            if ((changes.click_callback || changes.hover_callback) && !obj.interactions) {
                obj.geometry.boundsTree = new threeMeshBVH.MeshBVH(obj.geometry);
                obj.interactions = intersectCallback(obj, K3D);
            }

            if (!(changes.click_callback || changes.hover_callback) && obj.interactions) {
                obj.geometry.boundsTree = null;
                obj.interactions = null;
            }

            changes.click_callback = null;
            changes.hover_callback = null;
        }
    }
};