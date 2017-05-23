'use strict';
var Detector = require('./../../../../node_modules/three/examples/js/Detector');

/**
 * Get object By id
 * @param String} id
 * @method getObjectById
 * @memberof K3D.Providers.ThreeJS.Helpers
 * @returns {Object}
 */

function getSpaceDimensionsFromTargetElement(world) {
    return [world.targetDOMNode.offsetWidth, world.targetDOMNode.offsetHeight];
}

module.exports = {
    getObjectById: function (world, id) {
        return world.K3DObjects.getObjectByProperty('K3DIdentifier', id);
    },

    /**
     * Get dimensions of the space in which we're going to draw
     * @param {K3D.Core~world} world
     * @method getSpaceDimensionsFromTargetElement
     * @memberof K3D.Providers.ThreeJS.Helpers
     * @returns {Array.<Number,Number>}
     */
    getSpaceDimensionsFromTargetElement: getSpaceDimensionsFromTargetElement,

    /**
     * Window resize listener
     * @param {K3D.Core~world} world
     * @method resizeListener
     * @memberof K3D.Providers.ThreeJS.Helpers
     */
    resizeListener: function (world) {

        var dimensions = getSpaceDimensionsFromTargetElement(world);

        world.width = dimensions[0];
        world.height = dimensions[1];

        world.camera.aspect = world.width / world.height;
        world.camera.updateProjectionMatrix();

        world.renderer.setSize(world.width, world.height);
    },

    /**
     * getColorsArray
     * @memberof K3D.Providers.ThreeJS.Helpers
     * @param  {THREE.Color} color
     * @param  {Number} size
     * @return {Float32Array}
     */
    getColorsArray: function (color, size) {
        var colors = new Float32Array(size * 3),
            i;

        for (i = 0; i < colors.length; i += 3) {
            colors[i] = color.r;
            colors[i + 1] = color.g;
            colors[i + 2] = color.b;
        }

        return colors;
    },

    /**
     * getTwoColorsArray
     * @memberof K3D.Providers.ThreeJS.Helpers
     * @param  {THREE.Color} color
     * @param  {Number} size
     * @return {Float32Array}
     */
    getTwoColorsArray: function (color1, color2, size) {
        var colors = new Float32Array(size * 3),
            i;

        for (i = 0; i < colors.length; i += 6) {
            colors[i] = color1.r;
            colors[i + 1] = color1.g;
            colors[i + 2] = color1.b;

            colors[i + 3] = color2.r;
            colors[i + 4] = color2.g;
            colors[i + 5] = color2.b;
        }

        return colors;
    },

    /**
     * generateArrow
     * @memberof K3D.Providers.ThreeJS.Helpers
     * @param  {THREE.Geometry} coneGeometry
     * @param  {Array} lineVertices
     * @param  {THREE.Geometry} heads
     * @param  {THREE.Vector3d} origin
     * @param  {THREE.Vector3d} destination
     * @param  {THREE.Color} headColor
     */
    generateArrow: function (coneGeometry, lineVertices, heads, origin,
                             destination, headColor) {

        var axis = new THREE.Vector3(),
            direction = new THREE.Vector3().subVectors(destination, origin).normalize(),
            quaternion = new THREE.Quaternion();

        // line
        lineVertices.push(origin.x, origin.y, origin.z);
        lineVertices.push(destination.x, destination.y, destination.z);

        // head
        if (coneGeometry !== null) {
            coneGeometry.faces.forEach(function (face) {
                face.vertexColors.push(headColor, headColor, headColor);
            });

            if (direction.y > 0.99999) {
                quaternion.set(0, 0, 0, 1);
            } else if (direction.y < -0.99999) {
                quaternion.set(1, 0, 0, 0);
            } else {
                axis.set(direction.z, 0, -direction.x).normalize();
                quaternion = new THREE.Quaternion().setFromAxisAngle(axis, Math.acos(direction.y));
            }

            coneGeometry.applyMatrix(new THREE.Matrix4().makeRotationFromQuaternion(quaternion));

            coneGeometry.translate(destination.x, destination.y, destination.z);

            if (!heads) {
                heads = coneGeometry;
            } else {
                heads.merge(coneGeometry);
            }
        }

        return heads;
    },

    validateWebGL: function (targetDOMNode) {
        if (!Detector.webgl) {
            Detector.addGetWebGLMessage({parent: targetDOMNode});
            return false;
        }

        return true;
    }
};
