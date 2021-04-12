'use strict';

var THREE = require('three'),
    intersectHelper = require('./../helpers/Intersection'),
    handleColorMap = require('./../helpers/Fn').handleColorMap,
    areAllChangesResolve = require('./../helpers/Fn').areAllChangesResolve,
    commonUpdate = require('./../helpers/Fn').commonUpdate,
    getSide = require('./../helpers/Fn').getSide,
    buffer = require('./../../../core/lib/helpers/buffer');

/**
 * Loader strategy to handle Mesh object
 * @method Mesh
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config, K3D) {
        return new Promise(function (resolve) {
            config.color = typeof (config.color) !== 'undefined' ? config.color : 255;
            config.wireframe = typeof (config.wireframe) !== 'undefined' ? config.wireframe : false;
            config.flat_shading = typeof (config.flat_shading) !== 'undefined' ? config.flat_shading : true;
            config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;

            var modelMatrix = new THREE.Matrix4(),
                MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial,
                material,
                texture = new THREE.Texture(),
                textureImage = config.texture,
                textureFileFormat = config.texture_file_format,
                colors = (config.colors && config.colors.data) || null,
                colorRange = config.color_range,
                colorMap = (config.color_map && config.color_map.data) || null,
                attribute = (config.attribute && config.attribute.data) || null,
                triangleAttribute = (config.triangles_attribute && config.triangles_attribute.data) || null,
                vertices = (config.vertices && config.vertices.data) || null,
                indices = (config.indices && config.indices.data) || null,
                uvs = (config.uvs && config.uvs.data) || null,
                geometry = new THREE.BufferGeometry(),
                ObjectConstructor = THREE.Mesh,
                image,
                object,
                preparedtriangleAttribute, i;

            modelMatrix.set.apply(modelMatrix, config.model_matrix.data);

            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            geometry.setIndex(new THREE.BufferAttribute(indices, 1));

            material = new MaterialConstructor({
                color: config.color,
                emissive: 0,
                shininess: 50,
                specular: 0x111111,
                side: getSide(config),
                flatShading: config.flat_shading,
                wireframe: config.wireframe,
                opacity: config.opacity,
                depthWrite: config.opacity === 1.0,
                transparent: config.opacity !== 1.0
            });

            function finish() {
                if (config.flat_shading === false) {
                    geometry.computeVertexNormals();
                }

                geometry.computeBoundingSphere();
                geometry.computeBoundingBox();

                object = new ObjectConstructor(geometry, material);

                intersectHelper.init(config, object, K3D);

                object.applyMatrix4(modelMatrix);
                object.updateMatrixWorld();

                resolve(object);
            }

            if (colors !== null && colors.length > 0) {
                material.setValues({
                    color: 0xffffff,
                    vertexColors: THREE.VertexColors
                });

                geometry.setAttribute('color', new THREE.BufferAttribute(buffer.colorsToFloat32Array(colors), 3));
                finish();
            } else if (
                attribute && colorRange && colorMap && attribute.length > 0 &&
                colorRange.length > 0 && colorMap.length > 0
            ) {
                handleColorMap(geometry, colorMap, colorRange, attribute, material);
                finish();
            } else if (
                triangleAttribute && colorRange && colorMap && triangleAttribute.length > 0 &&
                colorRange.length > 0 && colorMap.length > 0
            ) {
                geometry = geometry.toNonIndexed();
                preparedtriangleAttribute = new Float32Array(triangleAttribute.length * 3);

                for (i = 0; i < preparedtriangleAttribute.length; i++) {
                    preparedtriangleAttribute[i] = triangleAttribute[Math.floor(i / 3)];
                }

                handleColorMap(geometry, colorMap, colorRange, preparedtriangleAttribute, material);
                finish();
            } else if (textureImage && textureFileFormat && uvs) {
                image = document.createElement('img');
                image.src = 'data:image/' + textureFileFormat + ';base64,' +
                    buffer.bufferToBase64(textureImage.buffer);

                geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));

                image.onload = function () {
                    material.map = texture;
                    texture.image = image;
                    texture.flipY = false;
                    texture.minFilter = THREE.LinearFilter;
                    texture.needsUpdate = true;
                    material.needsUpdate = true;
                    finish();
                };
            } else {
                finish();
            }
        });
    },

    update: function (config, changes, obj, K3D) {
        var resolvedChanges = {}, data, i;

        if (typeof (obj.geometry.attributes.uv) !== 'undefined') {
            if (typeof (changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
                data = obj.geometry.attributes.uv.array;

                if (config.attribute.data.length > 0) {
                    for (i = 0; i < data.length; i++) {
                        data[i] = (config.attribute.data[i] - config.color_range[0]) /
                            (config.color_range[1] - config.color_range[0]);
                    }
                }

                if (config.triangles_attribute.data.length > 0) {
                    for (i = 0; i < data.length; i++) {
                        data[i] = (config.triangles_attribute.data[Math.floor(i / 3)] - config.color_range[0]) /
                            (config.color_range[1] - config.color_range[0]);
                    }
                }

                obj.geometry.attributes.uv.needsUpdate = true;
                resolvedChanges.color_range = null;
            }

            if (typeof (changes.attribute) !== 'undefined' && !changes.attribute.timeSeries) {
                data = obj.geometry.attributes.uv.array;

                for (i = 0; i < data.length; i++) {
                    data[i] = (changes.attribute.data[i] - config.color_range[0]) /
                        (config.color_range[1] - config.color_range[0]);
                }

                obj.geometry.attributes.uv.needsUpdate = true;
                resolvedChanges.attribute = null;
            }
        }

        if (typeof (changes.opacity) !== 'undefined' && !changes.opacity.timeSeries) {
            obj.material.opacity = changes.opacity;
            obj.material.depthWrite = changes.opacity === 1.0;
            obj.material.transparent = changes.opacity !== 1.0;
            obj.material.needsUpdate = true;

            resolvedChanges.opacity = null;
        }

        intersectHelper.update(config, changes, resolvedChanges, obj, K3D);

        commonUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
};
