'use strict';

var THREE = require('three'),
    colorMapHelper = require('./../../../core/lib/helpers/colorMap'),
    typedArrayToThree = require('./../helpers/Fn').typedArrayToThree;

/**
 * Loader strategy to handle Texture object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config, K3D) {
        return new Promise(function (resolve) {
            var geometry = new THREE.PlaneBufferGeometry(1, 1),
                modelMatrix = new THREE.Matrix4(),
                colorMap = (config.color_map && config.color_map.data) || null,
                colorRange = config.color_range,
                object,
                texture;

            texture = new THREE.DataTexture(config.attribute.data,
                config.attribute.shape[1], config.attribute.shape[0], THREE.RedFormat,
                typedArrayToThree(config.attribute.data.constructor));

            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;
            texture.generateMipmaps = false;
            texture.anisotropy = K3D.getWorld().renderer.capabilities.getMaxAnisotropy();
            texture.needsUpdate = true;

            var canvas = colorMapHelper.createCanvasGradient(colorMap, 1024);
            var colormap = new THREE.CanvasTexture(canvas, THREE.UVMapping, THREE.ClampToEdgeWrapping,
                THREE.ClampToEdgeWrapping, THREE.NearestFilter, THREE.NearestFilter);
            colormap.needsUpdate = true;

            var uniforms = {
                low: {value: colorRange[0]},
                high: {value: colorRange[1]},
                map: {type: 't', value: texture},
                colormap: {type: 't', value: colormap}
            };

            var material = new THREE.ShaderMaterial({
                uniforms: uniforms,
                vertexShader: require('./shaders/Texture.vertex.glsl'),
                fragmentShader: require('./shaders/Texture.fragment.glsl'),
                side: THREE.DoubleSide,
                clipping: true
            });

            geometry.computeBoundingSphere();
            geometry.computeBoundingBox();

            object = new THREE.Mesh(geometry, material);
            modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
            object.applyMatrix(modelMatrix);
            object.updateMatrixWorld();

            object.onRemove = function () {
                object.material.uniforms.map.value.dispose();
                object.material.uniforms.map.value = undefined;
                object.material.uniforms.colormap.value.dispose();
                object.material.uniforms.colormap.value = undefined;
            };

            resolve(object);
        });
    }
};
