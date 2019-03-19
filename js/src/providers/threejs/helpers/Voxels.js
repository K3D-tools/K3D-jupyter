'use strict';

var MeshLine = require('./THREE.MeshLine'),
    yieldingLoop = require('./../../../core/lib/helpers/yieldingLoop'),
    voxelMeshGenerator = require('./../../../core/lib/helpers/voxelMeshGenerator'),
    interactionsVoxels = require('./../interactions/Voxels'),
    buffer = require('./../../../core/lib/helpers/buffer');

function getVoxelChunkObject(K3D, config, voxelSize, chunkStructure) {
    var geometry = new THREE.BufferGeometry(),
        voxelsChunkObject = new THREE.Object3D(),
        MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial,
        lineWidth,
        line,
        material;

    voxelsChunkObject.position.set(
        chunkStructure.offset[0] / (voxelSize.width),
        chunkStructure.offset[1] / (voxelSize.height),
        chunkStructure.offset[2] / (voxelSize.length)
    );
    voxelsChunkObject.updateMatrix();

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
            opacity: config.opacity,
            depthWrite: config.opacity === 1.0,
            transparent: config.opacity !== 1.0,
            side: THREE.DoubleSide,
            wireframe: config.wireframe
        })
    ));

    if (config.outlines && !config.wireframe) {
        lineWidth = new THREE.Matrix4().fromArray(config.model_matrix.data).getMaxScaleOnAxis() /
                    (Math.max(voxelSize.width, voxelSize.height, voxelSize.length) * 10);
        line = new MeshLine.MeshLine();
        material = new MeshLine.MeshLineMaterial({
            color: new THREE.Color(config.outlines_color),
            opacity: 0.75 * config.opacity,
            sizeAttenuation: true,
            depthWrite: false,
            transparent: true,
            lineWidth: lineWidth,
            resolution: new THREE.Vector2(K3D.getWorld().width, K3D.getWorld().height),
            side: THREE.DoubleSide
        });

        line.setGeometry(new Float32Array(chunkStructure.outlines), true);
        line.geometry.computeBoundingSphere();
        line.geometry.computeBoundingBox();

        voxelsChunkObject.add(new THREE.Mesh(line.geometry, material));
    }

    return voxelsChunkObject;
}

module.exports = {
    create: function (config, voxelsChunks, size, K3D) {
        return new Promise(function (resolve) {
            var modelMatrix, colorMap,
                object = new THREE.Group(),
                generate,
                voxelChunkObject,
                rollOverMesh = new THREE.Mesh(
                    new THREE.BoxGeometry(1.2 / size[0], 1.2 / size[1], 1.2 / size[2])
                        .translate(0.5 / size[0], 0.5 / size[1], 0.5 / size[2]),
                    new THREE.MeshBasicMaterial({color: 0xff0000, opacity: 0.5, transparent: true})
                ),
                colorsToFloat32Array = buffer.colorsToFloat32Array,
                viewModelistenerId,
                resizelistenerId;

            config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;
            config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
            config.wireframe = typeof (config.wireframe) !== 'undefined' ? config.wireframe : false;
            config.outlines = typeof (config.outlines) !== 'undefined' ? config.outlines : true;
            config.outlines_color = typeof (config.outlines_color) !== 'undefined' ? config.outlines_color : 0;

            modelMatrix = new THREE.Matrix4().fromArray(config.model_matrix.data);
            colorMap = config.color_map.data || [16711680, 65280, 255, 16776960, 16711935, 65535];
            colorMap = colorsToFloat32Array(colorMap);

            object.voxelSize = {width: size[0], height: size[1], length: size[2]};

            yieldingLoop(voxelsChunks.length, 20, function (index) {
                var chunk = voxelsChunks[index];

                generate = voxelMeshGenerator.initializeGreedyVoxelMesh(
                    chunk,
                    colorMap,
                    object.voxelSize,
                    config.outlines,
                    config.opacity < 1.0
                );

                voxelChunkObject = getVoxelChunkObject(K3D, config, object.voxelSize, generate());

                voxelChunkObject.voxel = {
                    generate: generate,
                    chunk: chunk,
                    getVoxelChunkObject: getVoxelChunkObject.bind(this, K3D, config, object.voxelSize)
                };

                voxelChunkObject.children[0].interactions =
                    interactionsVoxels(object, voxelChunkObject, rollOverMesh, K3D);

                object.add(voxelChunkObject);
            }, function () {
                object.position.set(-0.5, -0.5, -0.5);
                object.updateMatrix();

                modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
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
    },

    generateRegularChunks: function (chunkSize, shape, voxels) {
        var chunkList = [],
            sizeX = Math.ceil(shape[2] / chunkSize),
            sizeY = Math.ceil(shape[1] / chunkSize),
            sizeZ = Math.ceil(shape[0] / chunkSize),
            x, y, z;


        for (x = 0; x < sizeX; x++) {
            for (y = 0; y < sizeY; y++) {
                for (z = 0; z < sizeZ; z++) {
                    chunkList.push({
                        voxels: voxels,
                        size: [
                            shape[2] - (x + 1) * chunkSize > 0 ? chunkSize : shape[2] - x * chunkSize,
                            shape[1] - (y + 1) * chunkSize > 0 ? chunkSize : shape[1] - y * chunkSize,
                            shape[0] - (z + 1) * chunkSize > 0 ? chunkSize : shape[0] - z * chunkSize
                        ],
                        offset: [x * chunkSize, y * chunkSize, z * chunkSize],
                        multiple: 1
                    });
                }
            }
        }

        return chunkList;
    }
};
