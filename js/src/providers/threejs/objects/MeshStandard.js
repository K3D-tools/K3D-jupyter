const THREE = require('three');
const interactionsHelper = require('../helpers/Interactions');
const colorMapHelper = require('../../../core/lib/helpers/colorMap');
const { handleColorMap } = require('../helpers/Fn');
const { areAllChangesResolve } = require('../helpers/Fn');
const { commonUpdate } = require('../helpers/Fn');
const { getSide } = require('../helpers/Fn');
const { guardIndices } = require('../helpers/Fn');
const { scaleToColorRange } = require('../helpers/Fn');
const { baseColor } = require('../helpers/Fn');
const buffer = require('../../../core/lib/helpers/buffer');

const maximumSlicePlanes = 8;

const WRAPPING = {
    clamp: THREE.ClampToEdgeWrapping,
    repeat: THREE.RepeatWrapping,
    mirror: THREE.MirroredRepeatWrapping,
};

// image traits and the material slots they fill; the metalness-roughness image fills two
const IMAGE_MAPS = [
    { trait: 'texture', slots: ['map'] },
    { trait: 'emissive_map', slots: ['emissiveMap'] },
    { trait: 'normal_map', slots: ['normalMap'] },
    { trait: 'metalness_roughness_map', slots: ['roughnessMap', 'metalnessMap'] },
    { trait: 'occlusion_map', slots: ['aoMap'] },
];

/**
 * Loader strategy to handle Mesh object
 * @method Mesh
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */

function getSlicePlanesUniform(slicePlanes, modelMatrix) {
    const planes = slicePlanes.slice(0, maximumSlicePlanes).map((p) => {
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

function hasData(field) {
    return Boolean(field && field.data && field.data.length > 0);
}

// blending, depth write and alpha test; the mask threshold scales with opacity, which fades on top
function alphaState(config, opacity) {
    const mode = config.alpha_mode || 'opaque';
    const blend = mode === 'blend' || opacity !== 1.0 || hasData(config.opacity_function);
    const cutoff = typeof (config.alpha_cutoff) !== 'undefined' ? config.alpha_cutoff : 0.5;

    return {
        transparent: blend,
        depthWrite: !blend,
        alphaTest: mode === 'mask' ? cutoff * opacity : 0,
    };
}

function applyAlphaState(material, config, opacity, K3D) {
    const state = alphaState(config, opacity);

    // three compiles the test in or out on its value being zero
    if ((material.alphaTest > 0) !== (state.alphaTest > 0)) {
        material.needsUpdate = true;
    }
    material.alphaTest = state.alphaTest;

    if (K3D.parameters.depthPeels === 0) {
        if (material.transparent !== state.transparent) {
            material.needsUpdate = true;
        }
        material.depthWrite = state.depthWrite;
        material.transparent = state.transparent;
    }
}

function loadImage(bytes, format) {
    return new Promise((resolve) => {
        const image = document.createElement('img');

        image.onload = () => resolve(image);
        image.onerror = () => resolve(null);
        image.src = `data:image/${format};base64,${buffer.bufferToBase64(bytes)}`;
    });
}

// every image map decoded, trait -> texture; one that fails is left out, not waited on
function loadImageMaps(config) {
    // one mode for both axes, or 'u v'
    const modes = (config.texture_wrap || 'clamp').split(' ');
    const wrapS = WRAPPING[modes[0]] || WRAPPING.clamp;
    const wrapT = WRAPPING[modes[modes.length - 1]] || WRAPPING.clamp;
    const loaded = {};

    return Promise.all(IMAGE_MAPS.map(({ trait }) => {
        if (!hasData(config[trait])) {
            return null;
        }

        const bytes = config[trait].data;
        const format = (trait === 'texture' && config.texture_file_format)
            || buffer.imageFormat(bytes);

        if (!format) {
            console.warn(`K3D: mesh ${trait} is not an image a browser decodes (PNG, JPEG, WebP, `
                + 'GIF) - the mesh is shown without it');
            return null;
        }

        return loadImage(bytes, format).then((image) => {
            if (image === null) {
                console.warn(`K3D: mesh ${trait} in ${format} could not be decoded `
                    + '- the mesh is shown without it');
                return;
            }

            const texture = new THREE.Texture(image);

            // glTF's convention: the first pixel row is the top of the image at v = 0
            texture.flipY = false;
            texture.wrapS = wrapS;
            texture.wrapT = wrapT;
            texture.generateMipmaps = true;
            texture.minFilter = THREE.LinearMipmapLinearFilter;
            texture.magFilter = THREE.LinearFilter;
            texture.needsUpdate = true;

            loaded[trait] = texture;
        });
    })).then(() => loaded);
}

// next to image maps the colormap cannot use `map` and `uv`: it gets a sampler and attribute of its own
function injectColorMap(material, geometry, colorMap, colorRange, values, opacityFunction) {
    const canvas = colorMapHelper.createCanvasGradient(colorMap, 1024, 1, opacityFunction);
    const texture = new THREE.CanvasTexture(
        canvas,
        THREE.UVMapping,
        THREE.ClampToEdgeWrapping,
        THREE.ClampToEdgeWrapping,
        THREE.NearestFilter,
        THREE.NearestFilter,
    );
    const scaled = new Float32Array(values.length);

    for (let i = 0; i < values.length; i++) {
        scaled[i] = scaleToColorRange(values[i], colorRange[0], colorRange[1]);
    }

    texture.needsUpdate = true;
    geometry.setAttribute('k3dColorMapValue', new THREE.BufferAttribute(scaled, 1));

    const uniform = { value: texture };
    const previous = material.onBeforeCompile;

    material.userData.k3dColorMap = uniform;
    material.onBeforeCompile = function (shader, renderer) {
        shader.uniforms.k3dColorMap = uniform;
        shader.vertexShader = `attribute float k3dColorMapValue;\nvarying float vK3dColorMapValue;\n${
            shader.vertexShader.replace(
                '#include <begin_vertex>',
                '#include <begin_vertex>\nvK3dColorMapValue = k3dColorMapValue;',
            )}`;
        shader.fragmentShader = `uniform sampler2D k3dColorMap;\nvarying float vK3dColorMapValue;\n${
            shader.fragmentShader.replace(
                '#include <map_fragment>',
                '#include <map_fragment>\n'
                + 'diffuseColor *= texture2D(k3dColorMap, vec2(vK3dColorMapValue, 0.5));',
            )}`;

        if (previous && previous !== THREE.Material.prototype.onBeforeCompile) {
            previous.call(this, shader, renderer);
        }
    };
    // three keys programs by onBeforeCompile.toString(), the same text for every closure
    material.customProgramCacheKey = () => `k3dColorMap:${material.userData.k3dPeelDepthOut ? 1 : 0}`;
}

// per-vertex colour and alpha as one attribute; 'opaque' ignores the alpha
function vertexColors(config, count) {
    const colors = hasData(config.colors) ? config.colors.data : null;
    const opacities = (config.alpha_mode || 'opaque') !== 'opaque' && hasData(config.opacities)
        && config.opacities.data.length === count ? config.opacities.data : null;

    if (colors === null && opacities === null) {
        return null;
    }

    if (opacities === null) {
        return new THREE.BufferAttribute(buffer.colorsToFloat32Array(colors), 3);
    }

    const rgba = new Float32Array(count * 4);

    for (let i = 0; i < count; i++) {
        const c = colors !== null ? colors[i] : 0xffffff;

        rgba[i * 4] = ((c >> 16) & 255) / 255;
        rgba[i * 4 + 1] = ((c >> 8) & 255) / 255;
        rgba[i * 4 + 2] = (c & 255) / 255;
        rgba[i * 4 + 3] = opacities[i];
    }

    return new THREE.BufferAttribute(rgba, 4);
}

function create(config, K3D) {
    config.color = typeof (config.color) !== 'undefined' ? config.color
        : baseColor(config, ['colors', 'attribute', 'triangles_attribute', 'texture'], 255);
    config.wireframe = typeof (config.wireframe) !== 'undefined' ? config.wireframe : false;
    config.flat_shading = typeof (config.flat_shading) !== 'undefined' ? config.flat_shading : true;
    config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;
    config.slice_planes = typeof (config.slice_planes) !== 'undefined' ? config.slice_planes : [];
    config.roughness = typeof (config.roughness) !== 'undefined' ? config.roughness : 0.4;
    config.metalness = typeof (config.metalness) !== 'undefined' ? config.metalness : 0.0;
    config.emissive = typeof (config.emissive) !== 'undefined' ? config.emissive : 0;
    config.emissive_intensity = typeof (config.emissive_intensity) !== 'undefined'
        ? config.emissive_intensity : 1.0;
    config.normal_scale = typeof (config.normal_scale) !== 'undefined' ? config.normal_scale : 1.0;
    config.occlusion_strength = typeof (config.occlusion_strength) !== 'undefined'
        ? config.occlusion_strength : 1.0;
    config.alpha_mode = config.alpha_mode || 'opaque';

    const modelMatrix = new THREE.Matrix4();
    const MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshStandardMaterial;
    const colorRange = config.color_range;
    const colorMap = (config.color_map && config.color_map.data) || null;
    const attribute = (config.attribute && config.attribute.data) || null;
    const opacityFunction = (config.opacity_function && config.opacity_function.data) || null;
    const triangleAttribute = (config.triangles_attribute && config.triangles_attribute.data) || null;
    const normals = (config.normals && config.normals.data) || null;
    const vertices = (config.vertices && config.vertices.data) || null;
    const indices = guardIndices((config.indices && config.indices.data) || null, vertices, 'mesh');
    const vertexCount = vertices ? Math.floor(vertices.length / 3) : 0;
    const uvs = hasData(config.uvs) && config.uvs.data.length >= vertexCount * 2 ? config.uvs.data : null;
    const uvs2 = hasData(config.uvs2) && config.uvs2.data.length >= vertexCount * 2 ? config.uvs2.data : null;
    const imageMapsGiven = IMAGE_MAPS.some(({ trait }) => hasData(config[trait]));
    const imageMaps = imageMapsGiven && uvs !== null;
    let geometry = new THREE.BufferGeometry();
    let object;

    const hasNormals = (normals !== null && normals.length > 0);

    if (imageMapsGiven && uvs === null) {
        console.warn('K3D: mesh textures need uvs, one pair per vertex - the mesh is shown without them');
    }

    modelMatrix.set.apply(modelMatrix, config.model_matrix.data);

    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));

    if (config.flat_shading === false && hasNormals) {
        geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    }

    if (imageMaps) {
        geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));

        if (uvs2 !== null) {
            geometry.setAttribute('uv1', new THREE.BufferAttribute(new Float32Array(uvs2), 2));
        }
    }

    const material = new MaterialConstructor(config.wireframe ? {
        color: config.color,
        side: getSide(config),
        wireframe: true,
        opacity: config.opacity,
    } : {
        color: config.color,
        emissive: config.emissive,
        emissiveIntensity: config.emissive_intensity,
        roughness: config.roughness,
        metalness: config.metalness,
        side: getSide(config),
        flatShading: config.flat_shading,
        wireframe: false,
        opacity: config.opacity,
    });

    if (K3D.parameters.depthPeels !== 0) {
        material.blending = THREE.NoBlending;
        material.onBeforeCompile = K3D.colorOnBeforeCompile;
        material.userData.k3dPeelDepthOut = true;
    }

    applyAlphaState(material, config, config.opacity, K3D);

    // colours of every source multiply - base colour, vertex colours, colormap, texture
    const colorAttribute = vertexColors(config, vertexCount);

    if (colorAttribute !== null) {
        geometry.setAttribute('color', colorAttribute);
        material.vertexColors = true;
    }

    const vertexColorMapped = attribute && colorRange && colorMap && attribute.length > 0
        && colorRange.length > 0 && colorMap.length > 0;
    const triangleColorMapped = !vertexColorMapped
        && triangleAttribute && colorRange && colorMap && triangleAttribute.length > 0
        && colorRange.length > 0 && colorMap.length > 0;
    let colorMapValues = null;

    if (vertexColorMapped) {
        colorMapValues = attribute;
    } else if (triangleColorMapped) {
        geometry = geometry.toNonIndexed();
        colorMapValues = new Float32Array(triangleAttribute.length * 3);

        for (let i = 0; i < colorMapValues.length; i++) {
            colorMapValues[i] = triangleAttribute[Math.floor(i / 3)];
        }
    }

    if (colorMapValues !== null) {
        if (imageMaps) {
            injectColorMap(material, geometry, colorMap, colorRange, colorMapValues, opacityFunction);
        } else {
            handleColorMap(geometry, colorMap, colorRange, colorMapValues, material, opacityFunction);
            // handleColorMap whitens the base colour, which now multiplies the map
            material.color.set(config.color);
        }
    }

    function finish(maps) {
        Object.keys(maps).forEach((trait) => {
            IMAGE_MAPS.find((m) => m.trait === trait).slots.forEach((slot) => {
                if (slot in material) {
                    material[slot] = maps[trait];
                }
            });
        });

        if (maps.normal_map && material.normalScale) {
            // derivative tangents flip green, as in three's GLTFLoader
            material.normalScale.set(config.normal_scale, -config.normal_scale);
        }

        if (maps.occlusion_map) {
            material.aoMapIntensity = config.occlusion_strength;
            // uvs2 when there is a second set; occlusion is the one map that commonly has it
            maps.occlusion_map.channel = uvs2 !== null ? 1 : 0;
        }

        material.needsUpdate = true;

        if (config.flat_shading === false && !hasNormals) {
            geometry.computeVertexNormals();
        }

        if (config.slice_planes && config.slice_planes.length > 0) {
            if (config.slice_planes.length > maximumSlicePlanes) {
                console.warn(`K3D: slice_planes is limited to ${maximumSlicePlanes} on a `
                    + `mesh; the remaining ${config.slice_planes.length - maximumSlicePlanes} `
                    + 'are not applied');
            }

            // the outline is drawn by its own shader, which has one colour and no
            // colormap: saying so beats letting an attribute disappear quietly
            if (colorMap && attribute && attribute.length > 0) {
                console.warn('K3D.slice_planes: the section outline is drawn in the '
                    + 'object colour - its colormap is not applied to the outline');
            }

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
                            value: Math.min(config.slice_planes.length, maximumSlicePlanes),
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

            if (K3D.createAODepthMaterial && !config.wireframe) {
                // a cut-out occludes where solid; a glowing surface is spared the darkening
                const aoDepth = K3D.createAODepthMaterial(material, config);

                if (aoDepth !== null) {
                    object.userData.k3dAODepthMaterial = aoDepth;
                }
            }
        }

        interactionsHelper.init(config, object, K3D);

        object.applyMatrix4(modelMatrix);
        object.updateMatrixWorld();

        return object;
    }

    if (imageMaps) {
        return loadImageMaps(config).then(finish);
    }

    return Promise.resolve(finish({}));
}

module.exports = {
    create(config, K3D) {
        // a bad input rejects the promise, as it did when all of this ran in its executor
        try {
            return create(config, K3D);
        } catch (e) {
            return Promise.reject(e);
        }
    },

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};
        let data;
        let i;

        if (!obj) {
            return false;
        }

        if (typeof (changes.vertices) !== 'undefined' && !changes.vertices.timeSeries
            && typeof (changes.indices) === 'undefined'
            && obj.geometry && obj.geometry.index !== null
            && obj.geometry.attributes.position.array.length === changes.vertices.data.length) {
            obj.geometry.attributes.position.array.set(changes.vertices.data);
            obj.geometry.attributes.position.needsUpdate = true;

            const userNormals = config.normals && config.normals.data && config.normals.data.length > 0;

            if (obj.geometry.attributes.normal && !userNormals) {
                obj.geometry.computeVertexNormals();
            }

            obj.geometry.computeBoundingSphere();
            obj.geometry.computeBoundingBox();

            resolvedChanges.vertices = null;
        }

        // a colormap injected next to image maps has no fast path; rebuild for its edits
        const injectedColorMap = Boolean(obj.material && obj.material.userData.k3dColorMap);

        if (!injectedColorMap && obj.geometry && typeof (obj.geometry.attributes.uv) !== 'undefined') {
            if (typeof (changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
                data = obj.geometry.attributes.uv.array;

                if (config.attribute.data.length > 0) {
                    for (i = 0; i < data.length; i++) {
                        data[i] = scaleToColorRange(
                            config.attribute.data[i],
                            config.color_range[0],
                            config.color_range[1],
                        );
                    }
                }

                if (config.triangles_attribute.data.length > 0) {
                    for (i = 0; i < data.length; i++) {
                        data[i] = scaleToColorRange(
                            config.triangles_attribute.data[Math.floor(i / 3)],
                            config.color_range[0],
                            config.color_range[1],
                        );
                    }
                }

                obj.geometry.attributes.uv.needsUpdate = true;
                resolvedChanges.color_range = null;
            }

            if (obj.geometry && typeof (changes.attribute) !== 'undefined' && !changes.attribute.timeSeries) {
                data = obj.geometry.attributes.uv.array;

                for (i = 0; i < data.length; i++) {
                    data[i] = scaleToColorRange(
                        changes.attribute.data[i],
                        config.color_range[0],
                        config.color_range[1],
                    );
                }

                obj.geometry.attributes.uv.needsUpdate = true;
                resolvedChanges.attribute = null;
            }

            const colorMapChanged = typeof (changes.color_map) !== 'undefined' && !changes.color_map.timeSeries;
            const opacityFunctionChanged = typeof (changes.opacity_function) !== 'undefined'
                && !changes.opacity_function.timeSeries;

            if (colorMapChanged || opacityFunctionChanged) {
                if (opacityFunctionChanged && obj.material.transparent === false) {
                    return false;
                }

                const canvas = colorMapHelper.createCanvasGradient(
                    (changes.color_map && changes.color_map.data) || config.color_map.data,
                    1024,
                    1,
                    (changes.opacity_function && changes.opacity_function.data)
                        || (config.opacity_function && config.opacity_function.data),
                );

                obj.material.map.image = canvas;
                obj.material.map.needsUpdate = true;
                obj.material.needsUpdate = true;

                resolvedChanges.color_map = null;
                resolvedChanges.opacity_function = null;
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

        if (typeof (changes.color) !== 'undefined' && !changes.color.timeSeries) {
            // the base colour multiplies every other colour source, so it always applies
            if (obj.material.color) {
                obj.material.color.set(changes.color);
            }
            if (obj.material.uniforms && obj.material.uniforms.diffuse) {
                obj.material.uniforms.diffuse.value = new THREE.Color(changes.color);
            }
            resolvedChanges.color = null;
        }

        if (typeof (changes.emissive) !== 'undefined' && !changes.emissive.timeSeries
            && obj.material.emissive) {
            obj.material.emissive.set(changes.emissive);
            resolvedChanges.emissive = null;
        }

        if (typeof (changes.emissive_intensity) !== 'undefined' && !changes.emissive_intensity.timeSeries
            && typeof (obj.material.emissiveIntensity) !== 'undefined') {
            obj.material.emissiveIntensity = changes.emissive_intensity;
            resolvedChanges.emissive_intensity = null;
        }

        if (typeof (changes.normal_scale) !== 'undefined' && !changes.normal_scale.timeSeries
            && obj.material.normalScale) {
            obj.material.normalScale.set(changes.normal_scale, -changes.normal_scale);
            resolvedChanges.normal_scale = null;
        }

        if (typeof (changes.occlusion_strength) !== 'undefined' && !changes.occlusion_strength.timeSeries
            && typeof (obj.material.aoMapIntensity) !== 'undefined') {
            obj.material.aoMapIntensity = changes.occlusion_strength;
            resolvedChanges.occlusion_strength = null;
        }

        // these rebuild the AO prepass material, so the object is rebuilt
        const aoDepthKeys = ['emissive', 'emissive_intensity', 'alpha_mode', 'alpha_cutoff'];

        if (obj.userData.k3dAODepthMaterial && aoDepthKeys.some((key) => typeof (changes[key]) !== 'undefined')) {
            return false;
        }

        if (typeof (changes.alpha_cutoff) !== 'undefined' && !changes.alpha_cutoff.timeSeries
            && obj.material.isMaterial && !obj.material.uniforms) {
            applyAlphaState(obj.material, config, config.opacity, K3D);
            resolvedChanges.alpha_cutoff = null;
        }

        // opacity before commonUpdate, which knows nothing of alpha_mode and the mask threshold
        if (typeof (changes.opacity) !== 'undefined' && !changes.opacity.timeSeries
            && obj.material.isMaterial && !obj.material.uniforms) {
            obj.material.opacity = changes.opacity;
            obj.material.side = getSide({ opacity: changes.opacity, side: config.side });
            applyAlphaState(obj.material, config, changes.opacity, K3D);
            obj.material.needsUpdate = true;
            resolvedChanges.opacity = null;
        }

        interactionsHelper.update(config, changes, resolvedChanges, obj);

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};
