'use strict';

var THREE = require('three'),
    BufferGeometryUtils = require('three/examples/jsm/utils/BufferGeometryUtils').BufferGeometryUtils,
    intersectHelper = require('./../helpers/Intersection'),
    marchingCubesPolygonise = require('./../../../core/lib/helpers/marchingCubesPolygonise'),
    yieldingLoop = require('./../../../core/lib/helpers/yieldingLoop'),
    areAllChangesResolve = require('./../helpers/Fn').areAllChangesResolve,
    commonUpdate = require('./../helpers/Fn').commonUpdate;

/**
 * Loader strategy to handle Marching Cubes object
 * @method MarchingCubes
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config, K3D) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 255;
        config.wireframe = typeof (config.wireframe) !== 'undefined' ? config.wireframe : false;
        config.flat_shading = typeof (config.flat_shading) !== 'undefined' ? config.flat_shading : true;
        config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;

        return new Promise(function (resolve) {
            var scalarField = config.scalar_field.data,
                width = config.scalar_field.shape[2],
                height = config.scalar_field.shape[1],
                length = config.scalar_field.shape[0],
                spacingsX = config.spacings_x,
                spacingsY = config.spacings_y,
                spacingsZ = config.spacings_z,
                isSpacings = false,
                level = config.level,
                modelMatrix = new THREE.Matrix4(),
                MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial,
                material = new MaterialConstructor({
                    color: config.color,
                    emissive: 0,
                    shininess: 50,
                    specular: 0x111111,
                    side: THREE.DoubleSide,
                    flatShading: config.flat_shading,
                    wireframe: config.wireframe,
                    opacity: config.opacity,
                    depthWrite: config.opacity === 1.0,
                    transparent: config.opacity !== 1.0
                }),
                geometry = new THREE.BufferGeometry(),
                positions = [],
                object,
                x, y, z = 0,
                j, k,
                polygonise = marchingCubesPolygonise;

            if (spacingsX && spacingsY && spacingsZ) {
                isSpacings = spacingsX.shape[0] === width - 1 && spacingsY.shape[0] === height - 1 &&
                    spacingsZ.shape[0] == length - 1;
            }

            var withoutSpacings = function (i) {
                var sx = 1.0 / (width - 1),
                    sy = 1.0 / (height - 1),
                    sz = 1.0 / (length - 1);

                y = 0;
                for (j = 0; j < height - 1; j++) {
                    x = 0;
                    for (k = 0; k < width - 1; k++) {
                        polygonise(positions, scalarField, level,
                            width, height, length,
                            k, j, i,
                            x, y, z,
                            sx, sy, sz
                        );
                        x += sx;
                    }
                    y += sy;
                }
                z += sz;
            };

            var withSpacings = function (i) {
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
                function () {
                    var sizeX = 1.0, sizeY = 1.0, sizeZ = 1.0;

                    positions = new Float32Array(positions);
                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                    if (config.flat_shading === false) {
                        geometry = BufferGeometryUtils.mergeVertices(geometry);
                        geometry.computeVertexNormals();
                    }

                    if (isSpacings) {
                        sizeX = spacingsX.data.reduce(function (p, v) {
                            return p + v;
                        }, 0);
                        sizeY = spacingsY.data.reduce(function (p, v) {
                            return p + v;
                        }, 0);
                        sizeZ = spacingsZ.data.reduce(function (p, v) {
                            return p + v;
                        }, 0);
                    }

                    geometry.boundingSphere = new THREE.Sphere(
                        new THREE.Vector3(0.5 * sizeX, 0.5 * sizeY, 0.5 * sizeZ),
                        new THREE.Vector3(0.5 * sizeX, 0.5 * sizeY, 0.5 * sizeZ).length()
                    );

                    geometry.boundingBox = new THREE.Box3(
                        new THREE.Vector3(0.0, 0.0, 0.0),
                        new THREE.Vector3(sizeX, sizeY, sizeZ)
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

    update: function (config, changes, obj, K3D) {
        var resolvedChanges = {};

        intersectHelper.update(config, changes, obj, K3D);

        if (typeof (changes.opacity) !== 'undefined' && !changes.opacity.timeSeries) {
            obj.material.opacity = changes.opacity;
            obj.material.depthWrite = changes.opacity === 1.0;
            obj.material.transparent = changes.opacity !== 1.0;
            obj.material.needsUpdate = true;

            resolvedChanges.opacity = null;
        }

        commonUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
};
