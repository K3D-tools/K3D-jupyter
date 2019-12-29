'use strict';

var THREE = require('three'),
    intersectHelper = require('./../helpers/Intersection'),
    colorMapHelper = require('./../../../core/lib/helpers/colorMap'),
    areAllChangesResolve = require('./../helpers/Fn').areAllChangesResolve,
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

            if (config.puv.data.length === 9) {
                var positionArray = geometry.attributes.position.array;

                var p = new THREE.Vector3().fromArray(config.puv.data, 0);
                var u = new THREE.Vector3().fromArray(config.puv.data, 3);
                var v = new THREE.Vector3().fromArray(config.puv.data, 6);

                p.toArray(positionArray, 0);
                p.clone().add(u).toArray(positionArray, 3);
                p.clone().add(v).toArray(positionArray, 6);
                p.clone().add(v).add(u).toArray(positionArray, 9);

                geometry.computeVertexNormals();
            }

            geometry.computeBoundingSphere();
            geometry.computeBoundingBox();

            object = new THREE.Mesh(geometry, material);

            intersectHelper.init(config, object, K3D);

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
    },

    update: function (config, changes, obj, K3D) {
        intersectHelper.update(config, changes, obj, K3D);

        if (areAllChangesResolve(changes)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
};
