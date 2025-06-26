const THREE = require('three');
const interactionsHelper = require('../helpers/Interactions');
const colorMapHelper = require('../../../core/lib/helpers/colorMap');
const {handleColorMap} = require('../helpers/Fn');
const {areAllChangesResolve} = require('../helpers/Fn');
const {commonUpdate} = require('../helpers/Fn');
const {getSide} = require('../helpers/Fn');
const buffer = require('../../../core/lib/helpers/buffer');

const maximumSlicePlanes = 8;

/**
 * Loader strategy to handle Mesh object
 * @method Mesh
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */

function getSlicePlanesUniform(slicePlanes, modelMatrix) {
    const planes = slicePlanes.map((p) => {
        const mathPlane = new THREE.Plane(new THREE.Vector3().fromArray(p), p[3]);
        const localPlane = mathPlane.clone().applyMatrix4((new THREE.Matrix4()).copy(modelMatrix).invert());

        return new THREE.Vector4().set(
            localPlane.normal.x,
            localPlane.normal.y,
            localPlane.normal.z,
            localPlane.constant,
        );
    });

    for (let i = planes.length; i < maximumSlicePlanes; i++) {
        planes[i] = new THREE.Vector4();
    }

    return planes;
}

module.exports = {
    create(config, K3D) {
        return new Promise((resolve) => {
            config.color = typeof (config.color) !== 'undefined' ? config.color : 255;
            config.wireframe = typeof (config.wireframe) !== 'undefined' ? config.wireframe : false;
            config.flat_shading = typeof (config.flat_shading) !== 'undefined' ? config.flat_shading : true;
            config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;
            config.slice_planes = typeof (config.slice_planes) !== 'undefined' ? config.slice_planes : [];
            config.shininess = typeof (config.shininess) !== 'undefined' ? config.shininess : 50.0;

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
            const normals = (config.normals && config.normals.data) || null;
            const vertices = (config.vertices && config.vertices.data) || null;
            const indices = (config.indices && config.indices.data) || null;
            const uvs = (config.uvs && config.uvs.data) || null;
            let geometry = new THREE.BufferGeometry();
            let image;
            let object;
            let preparedtriangleAttribute;

            const hasNormals = (normals !== null && normals.length > 0);

            modelMatrix.set.apply(modelMatrix, config.model_matrix.data);

            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            geometry.setIndex(new THREE.BufferAttribute(indices, 1));

            if (config.flat_shading === false && hasNormals) {
                geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
            }

            const material = new MaterialConstructor({
                color: config.color,
                emissive: 0,
                shininess: config.shininess,
                specular: 0x111111,
                side: getSide(config),
                flatShading: config.flat_shading,
                wireframe: config.wireframe,
                opacity: config.opacity,
            });

            if (K3D.parameters.depthPeels === 0) {
                material.depthWrite = config.opacity === 1.0;
                material.transparent = config.opacity !== 1.0;
            } else {
                material.blending = THREE.NoBlending;
                material.onBeforeCompile = K3D.colorOnBeforeCompile;
            }

            function finish() {
                if (config.flat_shading === false && !hasNormals) {
                    geometry.computeVertexNormals();
                }

                if (config.slice_planes && config.slice_planes.length > 0) {
                    geometry = geometry.toNonIndexed();
                    geometry.computeBoundingSphere();
                    geometry.computeBoundingBox();

                    const d = geometry.attributes.position.array;

                    const lines = new Float32Array((d.length / 3) * 2);
                    const next1 = new Float32Array((d.length / 3) * 2);
                    const next2 = new Float32Array((d.length / 3) * 2);

                    for (let i = 0; i < d.length / 9; i++) {
                        lines.set(d.slice(i * 9, i * 9 + 6), i * 6);

                        next1.set(
                            d.slice(i * 9 + 6, i * 9 + 9),
                            i * 6,
                        );
                        next2.set(
                            d.slice(i * 9 + 3, i * 9 + 6),
                            i * 6,
                        );

                        next1.set(
                            d.slice(i * 9 + 6, i * 9 + 9),
                            i * 6 + 3,
                        );
                        next2.set(
                            d.slice(i * 9, i * 9 + 3),
                            i * 6 + 3,
                        );
                    }
                    geometry.setAttribute('next1', new THREE.BufferAttribute(next1, 3));
                    geometry.setAttribute('next2', new THREE.BufferAttribute(next2, 3));
                    geometry.setAttribute('position', new THREE.BufferAttribute(lines, 3));

                    const sliceMaterial = new THREE.ShaderMaterial({
                        uniforms: THREE.UniformsUtils.merge([
                            THREE.UniformsLib.lights,
                            THREE.UniformsLib.common,
                            {
                                slicePlanes: {
                                    value: getSlicePlanesUniform(config.slice_planes, modelMatrix),
                                },
                                diffuse: {
                                    value: new THREE.Color(config.color),
                                },
                                slicePlanesCount: {
                                    value: config.slice_planes.length,
                                },
                                opacity: {
                                    value: config.opacity,
                                },
                            },
                        ]),
                        defines: {
                            MAXIMUM_SLICE_PLANES: maximumSlicePlanes,
                        },
                        color: config.color,
                        vertexShader: require('./shaders/MeshStandardSlice.vertex.glsl'),
                        fragmentShader: require('./shaders/MeshStandardSlice.fragment.glsl'),
                        depthWrite: config.opacity === 1.0,
                        transparent: config.opacity !== 1.0,
                        lights: true,
                        clipping: true,
                    });

                    object = new THREE.LineSegments(geometry, sliceMaterial);
                    object.renderOrder = 10;
                } else {
                    geometry.computeBoundingSphere();
                    geometry.computeBoundingBox();

                    object = new THREE.Mesh(geometry, material);
                    // if (config.wireframe) {
                    //     object = new THREE.LineSegments(geometry, material);
                    // } else {
                    //     object = new THREE.Mesh(geometry, material);
                    // }
                }

                interactionsHelper.init(config, object, K3D);

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
        let i;

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
                    1
                );

                obj.material.map.image = canvas;
                obj.material.map.needsUpdate = true;
                obj.material.needsUpdate = true;

                resolvedChanges.color_map = null;
            }
        }

        if (typeof (changes.slice_planes) !== 'undefined' && !changes.slice_planes.timeSeries) {
            if (changes.slice_planes.length === 0 && obj.material.uniforms) {
                return false;
            }

            if (changes.slice_planes.length !== 0 && !obj.material.uniforms) {
                return false;
            }

            if (changes.slice_planes.length === 0 && !obj.material.uniforms) {
                resolvedChanges.slice_planes = null;
            } else {
                obj.material.uniforms.slicePlanes.value = getSlicePlanesUniform(changes.slice_planes, obj.matrix);
                obj.material.uniforms.slicePlanesCount.value = changes.slice_planes.length;

                resolvedChanges.slice_planes = null;
            }
        }

        if (typeof (changes.model_matrix) !== 'undefined' && !changes.model_matrix.timeSeries) {
            if (config.slice_planes.length !== 0) {
                return false;
            }
        }

        interactionsHelper.update(config, changes, resolvedChanges, obj);

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj});
        }
        return false;
    },
};
