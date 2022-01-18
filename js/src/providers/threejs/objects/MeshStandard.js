const THREE = require('three');
const intersectHelper = require('../helpers/Intersection');
const colorMapHelper = require('../../../core/lib/helpers/colorMap');
const { handleColorMap } = require('../helpers/Fn');
const { areAllChangesResolve } = require('../helpers/Fn');
const { commonUpdate } = require('../helpers/Fn');
const { getSide } = require('../helpers/Fn');
const buffer = require('../../../core/lib/helpers/buffer');

/**
 * Loader strategy to handle Mesh object
 * @method Mesh
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config, K3D) {
        return new Promise((resolve) => {
            config.color = typeof (config.color) !== 'undefined' ? config.color : 255;
            config.wireframe = typeof (config.wireframe) !== 'undefined' ? config.wireframe : false;
            config.flat_shading = typeof (config.flat_shading) !== 'undefined' ? config.flat_shading : true;
            config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;

            const modelMatrix = new THREE.Matrix4();
            const MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial;
            const texture = new THREE.Texture();
            const textureImage = config.texture;
            const textureFileFormat = config.texture_file_format;
            const colors = (config.colors && config.colors.data) || null;
            const colorRange = config.color_range;
            const colorMap = (config.color_map && config.color_map.data) || null;
            const attribute = (config.attribute && config.attribute.data) || null;
            const triangleAttribute = (config.triangles_attribute && config.triangles_attribute.data) || null;
            const vertices = (config.vertices && config.vertices.data) || null;
            const indices = (config.indices && config.indices.data) || null;
            const uvs = (config.uvs && config.uvs.data) || null;
            let geometry = new THREE.BufferGeometry();
            let image;
            let object;
            let preparedtriangleAttribute;

            modelMatrix.set.apply(modelMatrix, config.model_matrix.data);

            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            geometry.setIndex(new THREE.BufferAttribute(indices, 1));

            const material = new MaterialConstructor({
                color: config.color,
                emissive: 0,
                shininess: 50,
                specular: 0x111111,
                side: getSide(config),
                flatShading: config.flat_shading,
                wireframe: config.wireframe,
                opacity: config.opacity,
                depthWrite: config.opacity === 1.0,
                transparent: config.opacity !== 1.0,
            });

            function finish() {
                if (config.flat_shading === false) {
                    geometry.computeVertexNormals();
                }

                geometry.computeBoundingSphere();
                geometry.computeBoundingBox();

                object = new THREE.Mesh(geometry, material);


                intersectHelper.init(config, object, K3D);

                object.applyMatrix4(modelMatrix);
                object.updateMatrixWorld();

                resolve(object);
            }

            if (colors !== null && colors.length > 0) {
                material.setValues({
                    color: 0xffffff,
                    vertexColors: THREE.VertexColors,
                });

                geometry.setAttribute('color', new THREE.BufferAttribute(buffer.colorsToFloat32Array(colors), 3));
                finish();
            } else if (
                attribute && colorRange && colorMap && attribute.length > 0
                && colorRange.length > 0 && colorMap.length > 0
            ) {
                handleColorMap(geometry, colorMap, colorRange, attribute, material);
                finish();
            } else if (
                triangleAttribute && colorRange && colorMap && triangleAttribute.length > 0
                && colorRange.length > 0 && colorMap.length > 0
            ) {
                geometry = geometry.toNonIndexed();
                preparedtriangleAttribute = new Float32Array(triangleAttribute.length * 3);

                for (let i = 0; i < preparedtriangleAttribute.length; i++) {
                    preparedtriangleAttribute[i] = triangleAttribute[Math.floor(i / 3)];
                }

                handleColorMap(geometry, colorMap, colorRange, preparedtriangleAttribute, material);
                finish();
            } else if (textureImage && textureFileFormat && uvs) {
                image = document.createElement('img');
                image.src = `data:image/${textureFileFormat};base64,${
                    buffer.bufferToBase64(textureImage.data.buffer)}`;

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

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};
        let data;
        let
            i;

        if (!obj) {
            return false;
        }

        if (obj.geometry && typeof (obj.geometry.attributes.uv) !== 'undefined') {
            if (typeof (changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
                data = obj.geometry.attributes.uv.array;

                if (config.attribute.data.length > 0) {
                    for (i = 0; i < data.length; i++) {
                        data[i] = (config.attribute.data[i] - config.color_range[0])
                            / (config.color_range[1] - config.color_range[0]);
                    }
                }

                if (config.triangles_attribute.data.length > 0) {
                    for (i = 0; i < data.length; i++) {
                        data[i] = (config.triangles_attribute.data[Math.floor(i / 3)] - config.color_range[0])
                            / (config.color_range[1] - config.color_range[0]);
                    }
                }

                obj.geometry.attributes.uv.needsUpdate = true;
                resolvedChanges.color_range = null;
            }

            if (obj.geometry && typeof (changes.attribute) !== 'undefined' && !changes.attribute.timeSeries) {
                data = obj.geometry.attributes.uv.array;

                for (i = 0; i < data.length; i++) {
                    data[i] = (changes.attribute.data[i] - config.color_range[0])
                        / (config.color_range[1] - config.color_range[0]);
                }

                obj.geometry.attributes.uv.needsUpdate = true;
                resolvedChanges.attribute = null;
            }

            if (typeof (changes.color_map) !== 'undefined' && !changes.color_map.timeSeries) {
                const canvas = colorMapHelper.createCanvasGradient(
                    (changes.color_map && changes.color_map.data) || config.color_map.data,
                    1024,
                );

                obj.material.map.image = canvas;
                obj.material.map.needsUpdate = true;
                obj.material.needsUpdate = true;

                resolvedChanges.color_map = null;
            }
        }

        if (typeof (changes.opacity) !== 'undefined' && !changes.opacity.timeSeries) {
            if (obj.material.uniforms && obj.material.uniforms.opacity) {
                obj.material.uniforms.opacity.value = changes.opacity;
            } else {
                obj.material.opacity = changes.opacity;
            }

            obj.material.depthWrite = changes.opacity === 1.0;
            obj.material.transparent = changes.opacity !== 1.0;
            obj.material.side = changes.opacity < 1.0 ? THREE.DoubleSide : THREE.FrontSide;

            resolvedChanges.opacity = null;
        }

        intersectHelper.update(config, changes, resolvedChanges, obj, K3D);

        commonUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};
