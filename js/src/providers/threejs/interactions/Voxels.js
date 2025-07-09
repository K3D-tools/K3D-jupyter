const THREE = require('three');
const { viewModes } = require('../../../core/lib/viewMode');
/**
 * Interactions handlers for Voxels object
 * @memberof K3D.Providers.ThreeJS.Interactions
 */
module.exports = function (object, mesh, rollOverMesh, K3D) {
    function updateObject(obj) {
        const newMesh = obj.voxel.getVoxelChunkObject(obj.voxel.generate());
        let i;

        for (i = 0; i < obj.children.length; i++) {
            obj.children[i].geometry = newMesh.children[i].geometry;
        }
    }

    function getVoxelCoordinate(intersect, outside) {
        const matrix = (new THREE.Matrix4()).copy(mesh.matrixWorld).invert();
        const normalMatrix = new THREE.Matrix3().getNormalMatrix(matrix);
        const point = intersect.point.clone().applyMatrix4(matrix);
        const normal = intersect.face.normal.clone().applyMatrix3(normalMatrix).normalize();
        const dir = outside ? 0.5 : -0.5;

        point.add(
            normal.multiplyScalar(
                dir / Math.max(object.voxelSize.width, object.voxelSize.height, object.voxelSize.length),
            ),
        );

        const x = Math.floor(point.x * object.voxelSize.width) + mesh.voxel.chunk.offset[0];
        const y = Math.floor(point.y * object.voxelSize.height) + mesh.voxel.chunk.offset[1];
        const z = Math.floor(point.z * object.voxelSize.length) + mesh.voxel.chunk.offset[2];

        if (x < 0 || y < 0 || z < 0
            || x >= object.voxelSize.width || y >= object.voxelSize.height || z >= object.voxelSize.length) {
            return false;
        }

        return {
            x,
            y,
            z,
        };
    }

    function findMesh(coordinate) {
        return object.children.find((obj) => {
            if (obj.voxel
                && coordinate.x >= obj.voxel.chunk.offset[0]
                && coordinate.x <= obj.voxel.chunk.offset[0] + obj.voxel.chunk.size[0]
                && coordinate.y >= obj.voxel.chunk.offset[1]
                && coordinate.y <= obj.voxel.chunk.offset[1] + obj.voxel.chunk.size[1]
                && coordinate.z >= obj.voxel.chunk.offset[2]
                && coordinate.z <= obj.voxel.chunk.offset[2] + obj.voxel.chunk.size[2]) {
                return obj;
            }

            return false;
        });
    }

    function updateChunk(voxelCoordinate, offset) {
        const nextMesh = findMesh({
            x: voxelCoordinate.x + offset.x,
            y: voxelCoordinate.y + offset.y,
            z: voxelCoordinate.z + offset.z,
        });

        if (nextMesh) {
            updateObject(nextMesh);
        }
    }

    function onHoverAdd(intersect) {
        const voxelCoordinate = getVoxelCoordinate(intersect, true);

        if (!voxelCoordinate) {
            return false;
        }

        rollOverMesh.visible = true;
        rollOverMesh.position.set(
            voxelCoordinate.x / object.voxelSize.width,
            voxelCoordinate.y / object.voxelSize.height,
            voxelCoordinate.z / object.voxelSize.length,
        );

        return true;
    }

    function onClickAdd(intersect) {
        const voxelCoordinate = getVoxelCoordinate(intersect, true);

        if (!voxelCoordinate) {
            return false;
        }

        const i = voxelCoordinate.x
            + voxelCoordinate.y * object.voxelSize.width
            + voxelCoordinate.z * object.voxelSize.width * object.voxelSize.height;

        if (mesh.voxel.chunk.voxels instanceof Uint8Array) {
            mesh.voxel.chunk.voxels[i] = K3D.parameters.voxelPaintColor;
        } else {
            mesh.voxel.chunk.voxels.set(
                voxelCoordinate.x,
                voxelCoordinate.y,
                voxelCoordinate.z,
                K3D.parameters.voxelPaintColor,
                true,
            );
        }

        updateObject(mesh);

        // we should handle case when voxelCoordinate is in another chunk
        const nextMesh = findMesh(voxelCoordinate);

        if (nextMesh && mesh.uuid !== nextMesh.uuid) {
            updateObject(nextMesh);
        }

        rollOverMesh.visible = false;

        return true;
    }

    function onHoverChange(intersect) {
        const voxelCoordinate = getVoxelCoordinate(intersect, false);

        if (!voxelCoordinate) {
            return false;
        }

        rollOverMesh.visible = true;
        rollOverMesh.position.set(
            voxelCoordinate.x / object.voxelSize.width,
            voxelCoordinate.y / object.voxelSize.height,
            voxelCoordinate.z / object.voxelSize.length,
        );

        return true;
    }

    function onClickChange(intersect) {
        const voxelCoordinate = getVoxelCoordinate(intersect, false);

        if (!voxelCoordinate) {
            return false;
        }

        const i = voxelCoordinate.x
            + voxelCoordinate.y * object.voxelSize.width
            + voxelCoordinate.z * object.voxelSize.width * object.voxelSize.height;

        if (mesh.voxel.chunk.voxels instanceof Uint8Array) {
            mesh.voxel.chunk.voxels[i] = K3D.parameters.voxelPaintColor;
        } else {
            mesh.voxel.chunk.voxels.set(
                voxelCoordinate.x,
                voxelCoordinate.y,
                voxelCoordinate.z,
                K3D.parameters.voxelPaintColor,
                true,
            );
        }

        updateObject(mesh);

        if (voxelCoordinate.x === mesh.voxel.chunk.offset.x) {
            updateChunk(voxelCoordinate, { x: -1, y: 0, z: 0 });
        }

        if (voxelCoordinate.x === mesh.voxel.chunk.offset.x + mesh.voxel.chunk.size - 1) {
            updateChunk(voxelCoordinate, { x: 1, y: 0, z: 0 });
        }

        if (voxelCoordinate.y === mesh.voxel.chunk.offset.y) {
            updateChunk(voxelCoordinate, { x: 0, y: -1, z: 0 });
        }

        if (voxelCoordinate.y === mesh.voxel.chunk.offset.y + mesh.voxel.chunk.size - 1) {
            updateChunk(voxelCoordinate, { x: 0, y: 1, z: 0 });
        }

        if (voxelCoordinate.z === mesh.voxel.chunk.offset.z) {
            updateChunk(voxelCoordinate, { x: 1, y: 0, z: -1 });
        }

        if (voxelCoordinate.z === mesh.voxel.chunk.offset.z + mesh.voxel.chunk.size - 1) {
            updateChunk(voxelCoordinate, { x: 0, y: 0, z: 1 });
        }

        rollOverMesh.visible = false;

        return true;
    }

    function onClickCallback(intersect) {
        const voxelCoordinate = getVoxelCoordinate(intersect, false);

        if (voxelCoordinate) {
            K3D.dispatch(K3D.events.VOXELS_CALLBACK, { coord: voxelCoordinate, object });
        }

        return false;
    }

    return {
        onHover(intersect, viewMode) {
            switch (viewMode) {
                case viewModes.add:
                    return onHoverAdd(intersect);
                case viewModes.change:
                case viewModes.callback:
                    return onHoverChange(intersect);
                default:
                    return false;
            }
        },
        onClick(intersect, viewMode) {
            switch (viewMode) {
                case viewModes.add:
                    return onClickAdd(intersect);
                case viewModes.change:
                    return onClickChange(intersect);
                case viewModes.callback:
                    return onClickCallback(intersect);
                default:
                    return false;
            }
        },
    };
};
