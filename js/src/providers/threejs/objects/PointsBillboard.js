'use strict';

var THREE = require('three'),
    buffer = require('./../../../core/lib/helpers/buffer'),
    Fn = require('./../helpers/Fn'),
    modelMatrixUpdate = Fn.modelMatrixUpdate,
    areAllChangesResolve = Fn.areAllChangesResolve,
    getColorsArray = Fn.getColorsArray;

/**
 * Loader strategy to handle Points object
 * @method Points
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config) {
        var modelMatrix = new THREE.Matrix4(),
            color = new THREE.Color(config.color),
            pointPositions = config.positions.data,
            pointColors = (config.colors && config.colors.data) || null,
            shader = config.shader,
            colors,
            object,
            material,
            colorsToFloat32Array = buffer.colorsToFloat32Array,
            fragmentShader,
            fragmentShaderMap = {
                'dot': require('./shaders/Points.dot.fragment.glsl'),
                'flat': require('./shaders/Points.flat.fragment.glsl'),
                '3d': require('./shaders/Points.3d.fragment.glsl'),
                '3dspecular': require('./shaders/Points.3d.fragment.glsl')
            },
            vertexShader,
            vertexShaderMap = {
                'dot': require('./shaders/Points.dot.vertex.glsl'),
                'flat': require('./shaders/Points.vertex.glsl'),
                '3d': require('./shaders/Points.vertex.glsl'),
                '3dspecular': require('./shaders/Points.vertex.glsl')
            };

        fragmentShader = fragmentShaderMap[shader.toLowerCase()] || fragmentShaderMap.flat;
        vertexShader = vertexShaderMap[shader.toLowerCase()] || vertexShaderMap.flat;

        material = new THREE.ShaderMaterial({
            uniforms: THREE.UniformsUtils.merge([
                THREE.UniformsLib.lights,
                THREE.UniformsLib.points
            ]),
            defines: {
                USE_SPECULAR: (shader === '3dSpecular' ? 1 : 0)
            },
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            opacity: config.opacity,
            depthTest: config.opacity === 1.0,
            transparent: config.opacity !== 1.0,
            lights: true,
            clipping: true,
            extensions: {
                fragDepth: true
            }
        });

        // monkey-patching for imitate THREE.PointsMaterial
        material.size = config.point_size;
        material.color = new THREE.Color(1.0, 1.0, 1.0);
        material.map = null;
        material.isPointsMaterial = true;

        colors = (pointColors && pointColors.length === pointPositions.length / 3 ?
                colorsToFloat32Array(pointColors) : getColorsArray(color, pointPositions.length / 3)
        );

        object = new THREE.Points(getGeometry(pointPositions, colors), material);

        Fn.expandBoundingBox(object.geometry.boundingBox, config.point_size * 0.5);

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },
    update: function (config, changes, obj) {
        if (typeof(changes.positions) !== 'undefined' && !changes.positions.timeSeries &&
            changes.positions.data.length === obj.geometry.attributes.positions.array.length) {
            obj.geometry.attributes.position.array.set(changes.positions.data);
            obj.geometry.attributes.position.needsUpdate = true;
            changes.positions = null;
        }

        modelMatrixUpdate(config, changes, obj);

        if (areAllChangesResolve(changes)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
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

    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    return geometry;
}
