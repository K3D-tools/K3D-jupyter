'use strict';
var THREE = require('three'),
    createCanvasGradient = require('./../../../core/lib/helpers/colorMap').createCanvasGradient,
    Float16Array = require('./../../../core/lib/helpers/float16Array');

function getSpaceDimensionsFromTargetElement(world) {
    return [world.targetDOMNode.offsetWidth, world.targetDOMNode.offsetHeight];
}

module.exports = {
    /**
     * Finds the nearest (greater than x) power of two of given x
     * @inner
     * @memberof K3D.Providers.ThreeJS.Objects.Text
     * @param {Number} x
     * @returns {Number}
     */
    closestPowOfTwo: function (x) {
        return Math.pow(2, Math.ceil(Math.log(x) / Math.log(2)));
    },

    /**
     * Get object By id
     * @param {K3D.Core} world
     * @param {String} id
     * @method getObjectById
     * @memberof K3D.Providers.ThreeJS.Helpers
     * @returns {Object}
     */
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
     * @param  {THREE.Color} color1
     * @param  {THREE.Color} color2
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
     * Increase dimensions of a bounding box by delta in all 6 directions
     * @memberof K3D.Providers.ThreeJS.Helpers
     * @param  {Box3} bbox
     * @param  {Number} delta
     */
    expandBoundingBox: function (bbox, delta) {
        bbox.min.x -= delta;
        bbox.max.x += delta;
        bbox.min.y -= delta;
        bbox.max.y += delta;
        bbox.min.z -= delta;
        bbox.max.z += delta;
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
     * @param  {Number} headHeight
     */
    generateArrow: function (coneGeometry, lineVertices, heads, origin,
                             destination, headColor, headHeight) {

        var axis = new THREE.Vector3(),
            direction = new THREE.Vector3().subVectors(destination, origin),
            length = direction.length(),
            quaternion = new THREE.Quaternion(),
            lineDestination;

        direction.normalize();

        // line
        lineVertices.push(origin.x, origin.y, origin.z);

        // head
        if (coneGeometry !== null) {
            lineDestination = new THREE.Vector3().copy(origin).add(
                direction.clone().multiplyScalar(length - headHeight)
            );

            lineVertices.push(lineDestination.x, lineDestination.y, lineDestination.z);

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
        } else {
            lineVertices.push(destination.x, destination.y, destination.z);
        }

        return heads;
    },

    handleColorMap: function (geometry, colorMap, colorRange, attributes, material) {
        var canvas, texture, uvs, i;

        canvas = createCanvasGradient(colorMap, 1024);

        texture = new THREE.CanvasTexture(canvas, THREE.UVMapping, THREE.ClampToEdgeWrapping,
            THREE.ClampToEdgeWrapping, THREE.NearestFilter, THREE.NearestFilter);
        texture.needsUpdate = true;

        material.setValues({
            map: texture,
            color: 0xffffff
        });

        if (attributes) {
            uvs = new Float32Array(attributes.length);

            for (i = 0; i < attributes.length; i++) {
                uvs[i] = (attributes[i] - colorRange[0]) / (colorRange[1] - colorRange[0]);
            }

            geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 1));
        }
    },

    typedArrayToThree: function (creator) {
        if (creator === Int16Array) {
            return THREE.ShortType;
        }

        if (creator === Int32Array) {
            return THREE.IntType;
        }

        if (creator === Float16Array) {
            return THREE.HalfFloatType;
        }

        if (creator === Float32Array) {
            return THREE.FloatType;
        }
    },

    areAllChangesResolve: function (changes) {
        return Object.keys(changes).every(function (key) {
            return changes[key] === null;
        });
    },

    recalculateFrustum: function (camera) {
        camera.frustum.setFromMatrix(new THREE.Matrix4().multiplyMatrices(
            camera.projectionMatrix, camera.matrixWorldInverse
        ));
    },

    modelMatrixUpdate: function (config, changes, obj) {
        if (typeof(changes.model_matrix) !== 'undefined' && !changes.model_matrix.timeSeries) {
            var modelMatrix = new THREE.Matrix4();

            modelMatrix.set.apply(modelMatrix, changes.model_matrix.data);
            obj.position.set(0.0, 0.0, 0.0);
            obj.rotation.set(0.0, 0.0, 0.0);
            obj.scale.set(1.0, 1.0, 1.0);

            obj.applyMatrix(modelMatrix);
            obj.updateMatrixWorld();

            changes.model_matrix = null;
        }
    }
};
