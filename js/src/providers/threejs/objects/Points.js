'use strict';

var buffer = require('./../../../core/lib/helpers/buffer'),
    getColorsArray = require('./../helpers/Fn').getColorsArray;
/**
 * Loader strategy to handle Points object
 * @method Points
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {K3D.Config} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var modelViewMatrix = new THREE.Matrix4(),
        color = new THREE.Color(config.get('color', 0)),
        material = new THREE.PointsMaterial({
            size: config.get('pointSize'),
            sizeAttenuation: !!config.get('pointSizeAttenuation', false),
            vertexColors: THREE.VertexColors
        }),
        pointsPositions = config.get('pointsPositions'),
        pointsColors = config.get('pointsColors'),
        positions,
        colors,
        object,
        toFloat32Array = buffer.toFloat32Array,
        colorsToFloat32Array = buffer.colorsToFloat32Array;

    if (typeof (pointsPositions) === 'string') {
        pointsPositions = buffer.base64ToArrayBuffer(pointsPositions);
    }

    if (typeof (pointsColors) === 'string') {
        pointsColors = buffer.base64ToArrayBuffer(pointsColors);
    }

    positions = toFloat32Array(pointsPositions);
    colors = pointsColors ? colorsToFloat32Array(pointsColors) : getColorsArray(color, positions.length / 3);
    object = new THREE.Points(getGeometry(positions, colors), material);

    modelViewMatrix.set.apply(modelViewMatrix, config.get('modelViewMatrix'));
    object.applyMatrix(modelViewMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};

/**
 * Setup BufferGeometry
 * @inner
 * @memberof K3D.Providers.ThreeJS.Objects.Points
 * @param  {Float32Array} positions
 * @param  {Float32Array} colors
 * @return {THREE.BufferGeometry}
 */
function getGeometry(positions, colors) {
    var geometry = new THREE.BufferGeometry();

    geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.addAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeBoundingSphere();

    return geometry;
}
