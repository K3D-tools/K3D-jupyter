'use strict';

var lut = require('./../../../core/lib/helpers/lut');
var _ = require('lodash');

/**
 * Loader strategy to handle Volume object
 * @method Volume
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {
    var geometry = new THREE.BoxBufferGeometry(1, 1, 1),
        modelMatrix = new THREE.Matrix4(),
        colorMap = (config.color_map && config.color_map.data) || null,
        colorRange = config.color_range,
        samples = config.samples || 512.0,
        object,
        texture;

    texture = new THREE.Texture3D(
        new Float32Array(config.volume.data),
        config.volume.shape[2],
        config.volume.shape[1],
        config.volume.shape[0],
        THREE.RedFormat,
        THREE.FloatType);

    texture.generateMipmaps = false;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.needsUpdate = true;

    var canvas = lut(colorMap, 1024);
    var colormap = new THREE.CanvasTexture(canvas, THREE.UVMapping, THREE.ClampToEdgeWrapping,
        THREE.ClampToEdgeWrapping, THREE.NearestFilter, THREE.NearestFilter);
    colormap.needsUpdate = true;

    var uniforms = {
        low: {value: colorRange[0]},
        high: {value: colorRange[1]},
        samples_per_unit: {value: samples},
        volumeTexture: {type: 't', value: texture},
        colormap: {type: 't', value: colormap}
    };

    var material = new THREE.ShaderMaterial({
        uniforms: _.merge(
            uniforms,
            THREE.UniformsLib.lights
        ),
        defines: {
            USE_SPECULAR: 1
        },
        vertexShader: require('./shaders/Volume.vertex.glsl'),
        fragmentShader: require('./shaders/Volume.fragment.glsl'),
        side: THREE.BackSide,
        depthTest: false,
        lights: true,
        clipping: true,
        transparent: true
    });

    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    object = new THREE.Mesh(geometry, material);

    modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
    object.applyMatrix(modelMatrix);
    object.updateMatrixWorld();

    object.onRemove = function () {
        object.material.uniforms.volumeTexture.value.dispose();
        object.material.uniforms.volumeTexture.value = undefined;
        object.material.uniforms.colormap.value.dispose();
        object.material.uniforms.colormap.value = undefined;
    };

    return Promise.resolve(object);
};
