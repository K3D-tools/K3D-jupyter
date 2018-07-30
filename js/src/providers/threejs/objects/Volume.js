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
    config.samples = config.samples || 512.0;

    var geometry = new THREE.BoxBufferGeometry(1, 1, 1),
        modelMatrix = new THREE.Matrix4(),
        translation = new THREE.Vector3(),
        rotation = new THREE.Quaternion(),
        scale = new THREE.Vector3(),
        colorMap = (config.color_map && config.color_map.data) || null,
        colorRange = config.color_range,
        samples = config.samples,
        object,
        texture,
        jitterTexture;

    modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
    modelMatrix.decompose(translation, rotation, scale);

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

    jitterTexture = new THREE.DataTexture(
        new Uint8Array(_.range(32 * 32).map(function () {
            return 255.0 * Math.random();
        })),
        32, 32, THREE.RedFormat, THREE.UnsignedByteType);
    jitterTexture.minFilter = THREE.LinearFilter;
    jitterTexture.magFilter = THREE.LinearFilter;
    jitterTexture.wrapS = jitterTexture.wrapT = THREE.RepeatWrapping;
    jitterTexture.generateMipmaps = false;
    jitterTexture.needsUpdate = true;

    var canvas = lut(colorMap, 1024);
    var colormap = new THREE.CanvasTexture(canvas, THREE.UVMapping, THREE.ClampToEdgeWrapping,
        THREE.ClampToEdgeWrapping, THREE.NearestFilter, THREE.NearestFilter);
    colormap.needsUpdate = true;

    var uniforms = {
        low: {value: colorRange[0]},
        high: {value: colorRange[1]},
        samples: {value: samples},
        translation: {value: translation},
        rotation: {value: rotation},
        scale: {value: scale},
        volumeTexture: {type: 't', value: texture},
        colormap: {type: 't', value: colormap},
        jitterTexture: {type: 't', value: jitterTexture}
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
