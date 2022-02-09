const THREE = require('three');
const BufferGeometryUtils = require('three/examples/jsm/utils/BufferGeometryUtils');
const intersectHelper = require('../helpers/Intersection');
const marchingCubesPolygonise = require('../../../core/lib/helpers/marchingCubesPolygonise');
const yieldingLoop = require('../../../core/lib/helpers/yieldingLoop');
const { areAllChangesResolve } = require('../helpers/Fn');
const { commonUpdate } = require('../helpers/Fn');

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
            const material = new MaterialConstructor({
                color: config.color,
                emissive: 0,
                shininess: 50,
                specular: 0x111111,
                side: config.wireframe ? THREE.FrontSide : THREE.DoubleSide,
                flatShading: config.flat_shading,
                wireframe: config.wireframe,
                opacity: config.opacity,
                depthWrite: config.opacity === 1.0,
                transparent: config.opacity !== 1.0,
            });
            let geometry = new THREE.BufferGeometry();
            let positions = [];
            let object;
            let x; let y; let z = 0;
            let j; let k;
            const polygonise = marchingCubesPolygonise;

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
                    let sizeX = 1.0; let sizeY = 1.0; let
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

                    intersectHelper.init(config, object, K3D);

                    modelMatrix.set.apply(modelMatrix, config.model_matrix.data);

                    object.position.set(-0.5, -0.5, -0.5);
                    object.initialPosition = object.position.clone();
                    object.updateMatrix();

                    object.applyMatrix4(modelMatrix);
                    object.updateMatrixWorld();

                    resolve(object);
                });
        });
    },

    update(config, changes, obj) {
        const resolvedChanges = {};

        intersectHelper.update(config, changes, resolvedChanges, obj);

        if (typeof (changes.opacity) !== 'undefined' && !changes.opacity.timeSeries) {
            obj.material.opacity = changes.opacity;
            obj.material.depthWrite = changes.opacity === 1.0;
            obj.material.transparent = changes.opacity !== 1.0;
            obj.material.needsUpdate = true;

            resolvedChanges.opacity = null;
        }

        commonUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};
