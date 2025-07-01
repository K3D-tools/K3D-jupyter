const THREE = require('three');
const threeMeshBVH = require('three-mesh-bvh');
const MeshLine = require('./THREE.MeshLine')(THREE);
const yieldingLoop = require('../../../core/lib/helpers/yieldingLoop');
const voxelMeshGenerator = require('../../../core/lib/helpers/voxelMeshGenerator');
const interactionsVoxels = require('../interactions/Voxels');
const buffer = require('../../../core/lib/helpers/buffer');

function getVoxelChunkObject(K3D, config, voxelSize, chunkStructure) {
    const geometry = new THREE.BufferGeometry();
    const voxelsChunkObject = new THREE.Object3D();
    const MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial;
    let lineWidth;
    let line;
    let material;

    voxelsChunkObject.position.set(
        chunkStructure.offset[0] / (voxelSize.width),
        chunkStructure.offset[1] / (voxelSize.height),
        chunkStructure.offset[2] / (voxelSize.length),
    );
    voxelsChunkObject.updateMatrix();

    geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array(chunkStructure.vertices), 3),
    );
    geometry.setAttribute(
        'color',
        new THREE.BufferAttribute(new Float32Array(chunkStructure.colors), 3),
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
            wireframe: config.wireframe,
        }),
    ));

    if (config.outlines && !config.wireframe) {
        lineWidth = new THREE.Matrix4().fromArray(config.model_matrix.data).getMaxScaleOnAxis()
            / (Math.max(voxelSize.width, voxelSize.height, voxelSize.length) * 10);
        line = new MeshLine.MeshLine();
        material = new MeshLine.MeshLineMaterial({
            color: new THREE.Color(config.outlines_color),
            opacity: 0.75 * config.opacity,
            sizeAttenuation: true,
            depthWrite: false,
            transparent: true,
            lineWidth,
            resolution: new THREE.Vector2(K3D.getWorld().width, K3D.getWorld().height),
            side: THREE.DoubleSide,
        });

        material.userData = { outline: true };

        line.setGeometry(new Float32Array(chunkStructure.outlines), true);
        line.geometry.computeBoundingSphere();
        line.geometry.computeBoundingBox();

        voxelsChunkObject.add(new THREE.Mesh(line.geometry, material));
    }

    return voxelsChunkObject;
}

function rebuildChunk(object, forRebuild) {
    const idsMap = {};

    console.log(`K3D.Voxels rebuildChunk Count:${forRebuild.size}`);
    object.children.forEach((g) => {
        if (g.voxel) {
            idsMap[g.voxel.chunk.id] = g;
        }
    });

    forRebuild.forEach((id) => {
        const mesh = idsMap[id];
        let newMesh;

        if (mesh) {
            newMesh = mesh.voxel.getVoxelChunkObject(mesh.voxel.generate());

            for (let j = 0; j < mesh.children.length; j++) {
                mesh.children[j].geometry.dispose();
                mesh.children[j].geometry = newMesh.children[j].geometry;
            }
        }
    });

    forRebuild.clear();
}

module.exports = {
    create(config, voxelsChunks, size, K3D) {
        const forRebuild = new Set();

        return new Promise((resolve) => {
            let colorMap;
            const object = new THREE.Group();
            let voxelChunkObject;
            const rollOverMesh = new THREE.Mesh(
                new THREE.BoxGeometry(1.2 / size[0], 1.2 / size[1], 1.2 / size[2])
                    .translate(0.5 / size[0], 0.5 / size[1], 0.5 / size[2]),
                new THREE.MeshBasicMaterial({ color: 0xff0000, opacity: 0.5, transparent: true }),
            );
            const { colorsToFloat32Array } = buffer;
            let viewModelistenerId;
            let resizelistenerId;

            config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;
            config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
            config.wireframe = typeof (config.wireframe) !== 'undefined' ? config.wireframe : false;
            config.outlines = typeof (config.outlines) !== 'undefined' ? config.outlines : true;
            config.outlines_color = typeof (config.outlines_color) !== 'undefined' ? config.outlines_color : 0;

            const modelMatrix = new THREE.Matrix4().fromArray(config.model_matrix.data);
            colorMap = config.color_map.data || [16711680, 65280, 255, 16776960, 16711935, 65535];
            colorMap = colorsToFloat32Array(colorMap);

            object.holdRemeshing = config._hold_remeshing || false;
            object.voxelSize = { width: size[0], height: size[1], length: size[2] };

            object.rebuildChunk = function () {
                rebuildChunk(object, forRebuild);
                K3D.render();
            };

            object.updateChunk = function (chunk, skipNeighbours) {
                object.children.some((obj) => {
                    if (obj.voxel && obj.voxel.chunk.id === chunk.id) {
                        obj.voxel.chunk.voxels.data = chunk.voxels.data;

                        forRebuild.add(chunk.id);

                        if (!skipNeighbours) {
                            obj.voxel.chunk.voxels.neighbours.forEach((n) => {
                                forRebuild.add(n.id);
                            });
                        }

                        if (!object.holdRemeshing) {
                            object.rebuildChunk();
                        }

                        return true;
                    }

                    return false;
                });
            };

            object.voxelsChunks = voxelsChunks;
            object.addChunk = function (chunk) {
                const generate = voxelMeshGenerator.initializeGreedyVoxelMesh(
                    chunk,
                    colorMap,
                    object.voxelSize,
                    config.outlines,
                    config.opacity < 1.0,
                );
                let chunkData;

                if (object.holdRemeshing) {
                    chunkData = {
                        offset: chunk.offset,
                        vertices: [],
                        colors: [],
                        outlines: [],
                    };
                } else {
                    chunkData = generate();
                }

                voxelChunkObject = getVoxelChunkObject(K3D, config, object.voxelSize, chunkData);

                voxelChunkObject.voxel = {
                    generate,
                    chunk,
                    getVoxelChunkObject: getVoxelChunkObject.bind(this, K3D, config, object.voxelSize),
                };

                if (chunkData.vertices.length > 0) {
                    voxelChunkObject.children[0].geometry.boundsTree = new threeMeshBVH.MeshBVH(
                        voxelChunkObject.children[0].geometry,
                    );

                    // var helper = new threeMeshBVH.Visualizer(voxelChunkObject.children[0]);
                    // helper.depth = 40;
                    // helper.update();
                    // voxelChunkObject.children[0].add(helper);
                }

                voxelChunkObject.children[0].interactions = interactionsVoxels(
                    object,
                    voxelChunkObject,
                    rollOverMesh,
                    K3D,
                );

                object.add(voxelChunkObject);

                return voxelChunkObject;
            };

            yieldingLoop(voxelsChunks.length, 20, (index) => {
                object.addChunk(voxelsChunks[index]);
            }, () => {
                object.position.set(-0.5, -0.5, -0.5);
                object.initialPosition = object.position.clone();
                object.updateMatrix();

                modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
                object.applyMatrix4(modelMatrix);

                rollOverMesh.visible = false;
                rollOverMesh.geometry.computeBoundingSphere();
                rollOverMesh.geometry.computeBoundingBox();

                object.add(rollOverMesh);
                object.updateMatrixWorld();

                viewModelistenerId = K3D.on(K3D.events.VIEW_MODE_CHANGE, () => {
                    rollOverMesh.visible = false;
                });

                resizelistenerId = K3D.on(K3D.events.RESIZED, () => {
                    object.children.forEach((obj) => {
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

    generateRegularChunks(chunkSize, shape, voxels) {
        const chunkList = [];
        const sizeX = Math.ceil(shape[2] / chunkSize);
        const sizeY = Math.ceil(shape[1] / chunkSize);
        const sizeZ = Math.ceil(shape[0] / chunkSize);
        let x;
        let y;
        let z;
        let
            i;

        for (x = 0, i = 0; x < sizeX; x++) {
            for (y = 0; y < sizeY; y++) {
                for (z = 0; z < sizeZ; z++, i++) {
                    chunkList.push({
                        voxels,
                        size: [
                            shape[2] - (x + 1) * chunkSize > 0 ? chunkSize : shape[2] - x * chunkSize,
                            shape[1] - (y + 1) * chunkSize > 0 ? chunkSize : shape[1] - y * chunkSize,
                            shape[0] - (z + 1) * chunkSize > 0 ? chunkSize : shape[0] - z * chunkSize,
                        ],
                        offset: [x * chunkSize, y * chunkSize, z * chunkSize],
                        multiple: 1,
                        idx: i,
                    });
                }
            }
        }

        return chunkList;
    },
};
