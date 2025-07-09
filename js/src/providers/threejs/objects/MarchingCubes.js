const THREE = require('three');
const BufferGeometryUtils = require('three/examples/jsm/utils/BufferGeometryUtils');
const interactionsHelper = require('../helpers/Interactions');
const marchingCubesPolygonise = require('../../../core/lib/helpers/marchingCubesPolygonise');
const yieldingLoop = require('../../../core/lib/helpers/yieldingLoop');
const { areAllChangesResolve, getSide, typedArrayToThree } = require('../helpers/Fn');
const { commonUpdate } = require('../helpers/Fn');
const colorMapHelper = require('../../../core/lib/helpers/colorMap');
const _ = require('../../../lodash');

function isAttribute(config) {
    return config.attribute && config.attribute.data && config.attribute.data.length > 0
        && config.color_range && config.color_range.length > 0
        && config.color_map && config.color_map.data && config.color_map.data.length > 0;
}

/**
 * Loader strategy to handle Marching Cubes object
 * @method MarchingCubes
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config, K3D) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 255;
        config.wireframe = typeof (config.wireframe) !== 'undefined' ? config.wireframe : false;
        config.flat_shading = typeof (config.flat_shading) !== 'undefined' ? config.flat_shading : true;
        config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;
        config.shininess = typeof (config.shininess) !== 'undefined' ? config.shininess : 50.0;

        return new Promise((resolve) => {
            const scalarField = config.scalar_field.data;
            const width = config.scalar_field.shape[2];
            const height = config.scalar_field.shape[1];
            const length = config.scalar_field.shape[0];
            const spacingsX = config.spacings_x;
            const spacingsY = config.spacings_y;
            const spacingsZ = config.spacings_z;
            let isSpacings = false;
            const { level } = config;
            const modelMatrix = new THREE.Matrix4();
            const MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial;
            const colorRange = config.color_range;
            const colorMap = (config.color_map && config.color_map.data) || null;
            let opacityFunction = null;
            let material = new MaterialConstructor({
                color: config.color,
                emissive: 0,
                shininess: config.shininess,
                specular: 0x111111,
                side: config.wireframe ? THREE.FrontSide : THREE.DoubleSide,
                flatShading: config.flat_shading,
                wireframe: config.wireframe,
                opacity: config.opacity,
            });
            let geometry = new THREE.BufferGeometry();
            let positions = [];
            let object;
            let x;
            let y;
            let z = 0;
            let j;
            let k;
            const polygonise = marchingCubesPolygonise;

            if (isAttribute(config)) {
                if (config.opacity_function && config.opacity_function.data && config.opacity_function.data.length > 0) {
                    opacityFunction = config.opacity_function.data;
                }

                const canvas = colorMapHelper.createCanvasGradient(colorMap, 1024, 1, opacityFunction);
                const colormap = new THREE.CanvasTexture(
                    canvas,
                    THREE.UVMapping,
                    THREE.ClampToEdgeWrapping,
                    THREE.ClampToEdgeWrapping,
                    THREE.NearestFilter,
                    THREE.NearestFilter,
                );
                colormap.needsUpdate = true;

                const texture = new THREE.Data3DTexture(
                    config.attribute.data,
                    config.attribute.shape[2],
                    config.attribute.shape[1],
                    config.attribute.shape[0],
                );
                texture.format = THREE.RedFormat;
                texture.type = typedArrayToThree(config.attribute.data.constructor);

                texture.generateMipmaps = false;
                texture.minFilter = THREE.LinearFilter;
                texture.magFilter = THREE.LinearFilter;
                texture.wrapT = THREE.ClampToEdgeWrapping;
                texture.wrapS = THREE.ClampToEdgeWrapping;
                texture.needsUpdate = true;

                material = new THREE.ShaderMaterial({
                    uniforms: _.merge(
                        {
                            opacity: { value: config.opacity },
                            low: { value: colorRange[0] },
                            high: { value: colorRange[1] },
                            volumeTexture: { type: 't', value: texture },
                            colormap: { type: 't', value: colormap },
                            emissive: { type: 'v3', value: new THREE.Vector3(0, 0, 0) },
                            specular: { type: 'v3', value: new THREE.Vector3(0.04, 0.04, 0.04) },
                            shininess: { value: config.shininess },

                        },
                        THREE.UniformsLib.lights,
                    ),
                    defines: {
                        FLAT_SHADED: config.flat_shading
                    },
                    side: getSide(config),
                    vertexShader: require('./shaders/MarchingCubesVolume.vertex.glsl'),
                    fragmentShader: require('./shaders/MarchingCubesVolume.fragment.glsl'),
                    wireframe: config.wireframe,
                    flatShading: config.flat_shading,
                    lights: true,
                    clipping: true
                });
            }

            if (K3D.parameters.depthPeels === 0) {
                material.depthWrite = (config.opacity === 1.0 && opacityFunction === null);
                material.transparent = (config.opacity !== 1.0 || opacityFunction !== null);
            } else {
                material.blending = THREE.NoBlending;
                material.onBeforeCompile = K3D.colorOnBeforeCompile;
            }

            if (spacingsX && spacingsY && spacingsZ) {
                isSpacings = spacingsX.shape[0] === width - 1 && spacingsY.shape[0] === height - 1
                    && spacingsZ.shape[0] === length - 1;
            }

            const withoutSpacings = function (i) {
                const sx = 1.0 / (width - 1);
                const sy = 1.0 / (height - 1);
                const sz = 1.0 / (length - 1);

                y = 0;
                for (j = 0; j < height - 1; j++) {
                    x = 0;
                    for (k = 0; k < width - 1; k++) {
                        polygonise(positions, scalarField, level,
                            width, height, length,
                            k, j, i,
                            x, y, z,
                            sx, sy, sz);
                        x += sx;
                    }
                    y += sy;
                }
                z += sz;
            };

            const withSpacings = function (i) {
                y = 0;
                for (j = 0; j < height - 1; j++) {
                    x = 0;
                    for (k = 0; k < width - 1; k++) {
                        polygonise(positions, scalarField, level,
                            width, height, length,
                            k, j, i,
                            x, y, z,
                            spacingsX.data[k], spacingsY.data[j], spacingsZ.data[i]);

                        x += spacingsX.data[k];
                    }
                    y += spacingsY.data[j];
                }

                z += spacingsZ.data[i];
            };

            yieldingLoop(length - 1, 5, isSpacings ? withSpacings : withoutSpacings,
                () => {
                    let sizeX = 1.0;
                    let sizeY = 1.0;
                    let
                        sizeZ = 1.0;

                    positions = new Float32Array(positions);
                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                    if (config.flat_shading === false) {
                        geometry = BufferGeometryUtils.mergeVertices(geometry);
                        geometry.computeVertexNormals();
                    }

                    if (isSpacings) {
                        sizeX = spacingsX.data.reduce((p, v) => p + v, 0);
                        sizeY = spacingsY.data.reduce((p, v) => p + v, 0);
                        sizeZ = spacingsZ.data.reduce((p, v) => p + v, 0);
                    }

                    geometry.boundingSphere = new THREE.Sphere(
                        new THREE.Vector3(0.5 * sizeX, 0.5 * sizeY, 0.5 * sizeZ),
                        new THREE.Vector3(0.5 * sizeX, 0.5 * sizeY, 0.5 * sizeZ).length(),
                    );

                    geometry.boundingBox = new THREE.Box3(
                        new THREE.Vector3(0.0, 0.0, 0.0),
                        new THREE.Vector3(sizeX, sizeY, sizeZ),
                    );

                    object = new THREE.Mesh(geometry, material);
                    object.scale.set(1.0 / sizeX, 1.0 / sizeY, 1.0 / sizeZ);

                    interactionsHelper.init(config, object, K3D);

                    modelMatrix.set.apply(modelMatrix, config.model_matrix.data);

                    object.position.set(-0.5, -0.5, -0.5);
                    object.initialPosition = object.position.clone();
                    object.updateMatrix();

                    object.applyMatrix4(modelMatrix);
                    object.updateMatrixWorld();

                    resolve(object);
                },
            );
        });
    },

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        interactionsHelper.update(config, changes, resolvedChanges, obj);

        if (typeof (changes.attribute) !== 'undefined' && !changes.attribute.timeSeries) {
            if (obj.material.uniforms &&
                obj.material.uniforms.volumeTexture.value.image.data.constructor === changes.attribute.data.constructor
                && obj.material.uniforms.volumeTexture.value.image.width === changes.attribute.shape[2]
                && obj.material.uniforms.volumeTexture.value.image.height === changes.attribute.shape[1]
                && obj.material.uniforms.volumeTexture.value.image.depth === changes.attribute.shape[0]) {
                obj.material.uniforms.volumeTexture.value.image.data = changes.attribute.data;
                obj.material.uniforms.volumeTexture.value.needsUpdate = true;

                resolvedChanges.volume = null;
            }
        }

        if (obj.material.uniforms &&
            typeof (changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
            obj.material.uniforms.low.value = changes.color_range[0];
            obj.material.uniforms.high.value = changes.color_range[1];

            resolvedChanges.color_range = null;
        }

        if (obj.material.uniforms &&
            (typeof (changes.color_map) !== 'undefined' && !changes.color_map.timeSeries)
            || (typeof (changes.opacity_function) !== 'undefined' && !changes.opacity_function.timeSeries)) {
            if (!(changes.opacity_function && obj.material.transparent === false)) {
                const canvas = colorMapHelper.createCanvasGradient(
                    (changes.color_map && changes.color_map.data) || config.color_map.data,
                    1024,
                    1,
                    (changes.opacity_function && changes.opacity_function.data) || config.opacity_function.data,
                );

                obj.material.uniforms.colormap.value.image = canvas;
                obj.material.uniforms.colormap.value.needsUpdate = true;

                resolvedChanges.color_map = null;
                resolvedChanges.opacity_function = null;
            }
        }

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};
