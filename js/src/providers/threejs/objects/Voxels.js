'use strict';

var MeshLine = require('./../helpers/THREE.MeshLine'),
    voxelMeshGenerator = require('./../../../core/lib/helpers/voxelMeshGenerator'),
    yieldingLoop = require('./../../../core/lib/helpers/yieldingLoop'),
    buffer = require('./../../../core/lib/helpers/buffer'),
    interactionsVoxels = require('./../interactions/Voxels');

const chunkSize = 32;

/**
 * Loader strategy to handle Voxels object
 * @method Voxel
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @param {Object} K3D
 */
module.exports = function (config, K3D) {
    return new Promise(function (resolve) {

        config.visible = typeof(config.visible) !== 'undefined' ? config.visible : true;
        config.wireframe = typeof(config.wireframe) !== 'undefined' ? config.wireframe : false;
        config.outlines = typeof(config.outlines) !== 'undefined' ? config.outlines : true;
        config.outlines_color = typeof(config.outlines_color) !== 'undefined' ? config.outlines_color : 0;

        var modelMatrix = new THREE.Matrix4().fromArray(config.model_matrix.buffer),
            width = config.voxels.shape[2],
            height = config.voxels.shape[1],
            length = config.voxels.shape[0],
            voxels = config.voxels.buffer,
            colorMap = config.color_map.buffer || [16711680, 65280, 255, 16776960, 16711935, 65535],
            object = new THREE.Group(),
            generate,
            voxelChunkObject,
            chunksCount = {
                x: Math.ceil(width / chunkSize),
                y: Math.ceil(height / chunkSize),
                z: Math.ceil(length / chunkSize)
            },
            offsets,
            rollOverMesh = new THREE.Mesh(
                new THREE.BoxGeometry(1.2 / width, 1.2 / height, 1.2 / length)
                    .translate(0.5 / width, 0.5 / height, 0.5 / length),
                new THREE.MeshBasicMaterial({color: 0xff0000, opacity: 0.5, transparent: true})
            ),
            colorsToFloat32Array = buffer.colorsToFloat32Array,
            viewModelistenerId,
            resizelistenerId;

        colorMap = colorsToFloat32Array(colorMap);

        object.voxelSize = {width: width, height: height, length: length};
        object.voxels = voxels;

        yieldingLoop(chunksCount.z * chunksCount.y, 5, function (index) {
            var x,
                y = index % chunksCount.y,
                z = Math.floor(index / chunksCount.y);

            for (x = 0; x < chunksCount.x; x++) {

                offsets = {x: x * chunkSize, y: y * chunkSize, z: z * chunkSize};

                generate = voxelMeshGenerator.initializeGreedyVoxelMesh(
                    voxels,
                    colorMap,
                    chunkSize,
                    object.voxelSize,
                    offsets,
                    config.outlines
                );

                voxelChunkObject = getVoxelChunkObject(K3D, config, generate());

                voxelChunkObject.voxel = {
                    generate: generate,
                    offsets: offsets,
                    getVoxelChunkObject: getVoxelChunkObject.bind(this, K3D, config),
                    chunkSize: chunkSize
                };

                voxelChunkObject.children[0].interactions =
                    interactionsVoxels(object, voxelChunkObject, rollOverMesh, K3D);

                object.add(voxelChunkObject);
            }
        }, function () {
            object.position.set(-0.5, -0.5, -0.5);
            object.updateMatrix();

            modelMatrix.set.apply(modelMatrix, config.model_matrix.buffer);
            object.applyMatrix(modelMatrix);

            rollOverMesh.visible = false;
            rollOverMesh.geometry.computeBoundingSphere();
            rollOverMesh.geometry.computeBoundingBox();

            object.add(rollOverMesh);
            object.updateMatrixWorld();

            viewModelistenerId = K3D.on(K3D.events.VIEW_MODE_CHANGE, function () {
                rollOverMesh.visible = false;
            });

            resizelistenerId = K3D.on(K3D.events.RESIZED, function () {
                object.children.forEach(function (obj) {
                    if (obj.children[1]) {
                        // update outlines
                        obj.children[1].material.uniforms.resolution.value.x = K3D.getWorld().width;
                        obj.children[1].material.uniforms.resolution.value.y = K3D.getWorld().height;
                    }
                });
            });

            object.onRemove = function () {
                K3D.off(K3D.events.VIEW_MODE_CHANGE, viewModelistenerId);
                K3D.off(K3D.events.RESIZED, resizelistenerId);
            };

            resolve(object);
        });
    });
};

function getVoxelChunkObject(K3D, config, chunkStructure) {
    var geometry = new THREE.BufferGeometry(),
        voxelsChunkObject = new THREE.Object3D(),
        MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial;


    geometry.addAttribute('position',
        new THREE.BufferAttribute(new Float32Array(chunkStructure.vertices), 3)
    );
    geometry.addAttribute('color',
        new THREE.BufferAttribute(new Float32Array(chunkStructure.colors), 3)
    );

    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    voxelsChunkObject.add(new THREE.Mesh(
        geometry,
        new MaterialConstructor({
            vertexColors: THREE.VertexColors,
            flatShading: true,
            side: THREE.DoubleSide,
            wireframe: config.wireframe
        })
    ));

    if (config.outlines && !config.wireframe) {
        var lineWidth = new THREE.Matrix4().fromArray(config.model_matrix.buffer).getMaxScaleOnAxis() /
            (Math.max(config.voxels.shape[0], config.voxels.shape[1], config.voxels.shape[2]) * 10);
        var line = new MeshLine.MeshLine();
        var material = new MeshLine.MeshLineMaterial({
            color: new THREE.Color(config.outlines_color),
            opacity: 0.75,
            sizeAttenuation: true,
            transparent: true,
            lineWidth: lineWidth,
            resolution: new THREE.Vector2(K3D.getWorld().width, K3D.getWorld().height),
            side: THREE.DoubleSide
        });

        line.setGeometry(new Float32Array(chunkStructure.outlines), true);
        line.geometry.computeBoundingSphere();
        line.geometry.computeBoundingBox();

        var mesh = new THREE.Mesh(line.geometry, material);
        voxelsChunkObject.add(mesh);
    }

    return voxelsChunkObject;
}
