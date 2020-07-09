'use strict';

var THREE = require('three'),
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
                    depthTest: config.opacity === 1.0,
                    depthWrite: config.opacity === 1.0,
                    transparent: config.opacity !== 1.0
                }),
                geometry = new THREE.BufferGeometry(),
                positions = [],
                object,
                x, y,
                polygonise = marchingCubesPolygonise;

            yieldingLoop(length - 1, 5, function (z) {
                for (y = 0; y < height - 1; y++) {
                    for (x = 0; x < width - 1; x++) {
                        polygonise(positions, scalarField, width, height, length, level, x, y, z);
                    }
                }
            }, function () {

                positions = new Float32Array(positions);
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                geometry.boundingSphere = new THREE.Sphere(
                    new THREE.Vector3(0.5, 0.5, 0.5),
                    new THREE.Vector3(0.5, 0.5, 0.5).length()
                );

                geometry.boundingBox = new THREE.Box3(
                    new THREE.Vector3(0.0, 0.0, 0.0),
                    new THREE.Vector3(1.0, 1.0, 1.0)
                );

                if (config.flat_shading === false) {
                    var geo = new THREE.Geometry().fromBufferGeometry(geometry);
                    geo.mergeVertices();
                    geo.computeVertexNormals();
                    geometry.fromGeometry(geo);
                }

                object = new THREE.Mesh(geometry, material);

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

        if (typeof(changes.opacity) !== 'undefined' && !changes.opacity.timeSeries) {
            obj.material.opacity = changes.opacity;
            obj.material.depthTest = changes.opacity === 1.0;
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
