'use strict';

var buffer = require('./../../../core/lib/helpers/buffer'),
    getColorsArray = require('./../helpers/Fn').getColorsArray;
/**
 * Loader strategy to handle Points object
 * @method PointsMesh
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configuration params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {
    var modelMatrix = new THREE.Matrix4(),
        color = new THREE.Color(config.color || 65280),
        positions = config.positions.buffer,
        pointColors = (config.colors && config.colors.buffer) || null,
        colors,
        object,
        colorsToFloat32Array = buffer.colorsToFloat32Array,
        phongShader = THREE.ShaderLib.phong,
        material = new THREE.ShaderMaterial({
            uniforms: THREE.UniformsUtils.merge([phongShader.uniforms, {shininess: {value: 80}}]),
            vertexShader: require('./shaders/PointsMesh.vertex.glsl'),
            fragmentShader: phongShader.fragmentShader,
            lights: true,
            vertexColors: THREE.VertexColors
        }),
        sphereGeometry = new THREE.IcosahedronBufferGeometry(config.point_size * 0.5, 2),
        instancedGeometry = new THREE.InstancedBufferGeometry().copy(sphereGeometry),
        geometry = new THREE.BufferGeometry();


    colors = ( pointColors && pointColors.length ===  positions.length / 3
        ? colorsToFloat32Array(pointColors)
        : getColorsArray(color, positions.length / 3)
    );

    instancedGeometry.addAttribute('offset', new THREE.InstancedBufferAttribute(new Float32Array(positions), 3));
    instancedGeometry.addAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array(colors), 3));

    // boundingBox & boundingSphere
    geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.computeBoundingSphere();
    instancedGeometry.boundingSphere = geometry.boundingSphere.clone();
    geometry.computeBoundingBox();
    instancedGeometry.boundingBox = geometry.boundingBox.clone();


    object = new THREE.Mesh(instancedGeometry, material);

    modelMatrix.set.apply(modelMatrix, config.model_matrix.buffer);
    object.applyMatrix(modelMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
