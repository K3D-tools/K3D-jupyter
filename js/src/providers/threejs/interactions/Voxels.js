'use strict';

var viewModes = require('./../../../core/lib/viewMode').viewModes;
/**
 * Interactions handlers for Voxels object
 * @method Voxels
 * @memberof K3D.Providers.ThreeJS.Interactions
 */
module.exports = function (object, mesh, rollOverMesh, K3D) {

    function updateObject(mesh) {
        var newMesh = mesh.voxel.getVoxelChunkObject(mesh.voxel.generate()),
            i;

        for (i = 0; i < mesh.children.length; i++) {
            mesh.children[i].geometry = newMesh.children[i].geometry;
        }
    }

    function getVoxelCoordinate(intersect, outside) {
        var matrix = new THREE.Matrix4().getInverse(mesh.matrixWorld),
            normalMatrix = new THREE.Matrix3().getNormalMatrix(matrix),
            x, y, z,
            point = intersect.point.clone().applyMatrix4(matrix),
            normal = intersect.face.normal.clone().applyMatrix3(normalMatrix).normalize(),
            dir = outside ? 0.5 : -0.5;

        point.add(
            normal.multiplyScalar(
                dir / Math.max(object.voxelSize.width, object.voxelSize.height, object.voxelSize.length)
            )
        );

        x = Math.floor(point.x * object.voxelSize.width) + mesh.voxel.chunk.offset[0];
        y = Math.floor(point.y * object.voxelSize.height) + mesh.voxel.chunk.offset[1];
        z = Math.floor(point.z * object.voxelSize.length) + mesh.voxel.chunk.offset[2];

        if (x < 0 || y < 0 || z < 0 ||
            x >= object.voxelSize.width || y >= object.voxelSize.height || z >= object.voxelSize.length) {
            return false;
        }

        return {
            x: x,
            y: y,
            z: z
        };
    }

    function findMesh(coordinate) {
        return object.children.find(function (mesh) {
            if (mesh.voxel &&
                coordinate.x >= mesh.voxel.chunk.offset[0] &&
                coordinate.x <= mesh.voxel.chunk.offset[0] + mesh.voxel.chunk.size[0] &&
                coordinate.y >= mesh.voxel.chunk.offset[1] &&
                coordinate.y <= mesh.voxel.chunk.offset[1] + mesh.voxel.chunk.size[1] &&
                coordinate.z >= mesh.voxel.chunk.offset[2] &&
                coordinate.z <= mesh.voxel.chunk.offset[2] + mesh.voxel.chunk.size[2]) {

                return mesh;
            }
        });
    }

    function updateChunk(voxelCoordinate, offset) {
        var nextMesh = findMesh({
            x: voxelCoordinate.x + offset.x,
            y: voxelCoordinate.y + offset.y,
            z: voxelCoordinate.z + offset.z
        });

        if (nextMesh) {
            updateObject(nextMesh);
        }
    }

    function onHoverAdd(intersect) {
        var voxelCoordinate = getVoxelCoordinate(intersect, true);

        if (!voxelCoordinate) {
            return false;
        }

        rollOverMesh.visible = true;
        rollOverMesh.position.set(
            voxelCoordinate.x / object.voxelSize.width,
            voxelCoordinate.y / object.voxelSize.height,
            voxelCoordinate.z / object.voxelSize.length
        );

        return true;
    }

    function onClickAdd(intersect) {
        var voxelCoordinate = getVoxelCoordinate(intersect, true), nextMesh, i;

        if (!voxelCoordinate) {
            return false;
        }

        i = voxelCoordinate.x +
            voxelCoordinate.y * object.voxelSize.width +
            voxelCoordinate.z * object.voxelSize.width * object.voxelSize.height;

        if (mesh.voxel.chunk.voxels instanceof Uint8Array) {
            mesh.voxel.chunk.voxels[i] = K3D.parameters.voxelPaintColor;
        } else {
            mesh.voxel.chunk.voxels.set(voxelCoordinate.x, voxelCoordinate.y, voxelCoordinate.z,
                K3D.parameters.voxelPaintColor, true);
        }

        updateObject(mesh);

        // we should handle case when voxelCoordinate is in another chunk
        nextMesh = findMesh(voxelCoordinate);

        if (nextMesh && mesh.uuid !== nextMesh.uuid) {
            updateObject(nextMesh);
        }

        rollOverMesh.visible = false;

        return true;
    }

    function onHoverChange(intersect) {
        var voxelCoordinate = getVoxelCoordinate(intersect, false);

        if (!voxelCoordinate) {
            return false;
        }

        rollOverMesh.visible = true;
        rollOverMesh.position.set(
            voxelCoordinate.x / object.voxelSize.width,
            voxelCoordinate.y / object.voxelSize.height,
            voxelCoordinate.z / object.voxelSize.length
        );

        return true;
    }

    function onClickChange(intersect) {
        var voxelCoordinate = getVoxelCoordinate(intersect, false), i;

        if (!voxelCoordinate) {
            return false;
        }

        i = voxelCoordinate.x +
            voxelCoordinate.y * object.voxelSize.width +
            voxelCoordinate.z * object.voxelSize.width * object.voxelSize.height;

        if (mesh.voxel.chunk.voxels instanceof Uint8Array) {
            mesh.voxel.chunk.voxels[i] = K3D.parameters.voxelPaintColor;
        } else {
            mesh.voxel.chunk.voxels.set(voxelCoordinate.x, voxelCoordinate.y, voxelCoordinate.z,
                K3D.parameters.voxelPaintColor, true);
        }

        updateObject(mesh);

        if (voxelCoordinate.x === mesh.voxel.chunk.offset.x) {
            updateChunk(voxelCoordinate, {x: -1, y: 0, z: 0});
        }

        if (voxelCoordinate.x === mesh.voxel.chunk.offset.x + mesh.voxel.chunk.size - 1) {
            updateChunk(voxelCoordinate, {x: 1, y: 0, z: 0});
        }

        if (voxelCoordinate.y === mesh.voxel.chunk.offset.y) {
            updateChunk(voxelCoordinate, {x: 0, y: -1, z: 0});
        }

        if (voxelCoordinate.y === mesh.voxel.chunk.offset.y + mesh.voxel.chunk.size - 1) {
            updateChunk(voxelCoordinate, {x: 0, y: 1, z: 0});
        }

        if (voxelCoordinate.z === mesh.voxel.chunk.offset.z) {
            updateChunk(voxelCoordinate, {x: 1, y: 0, z: -1});
        }

        if (voxelCoordinate.z === mesh.voxel.chunk.offset.z + mesh.voxel.chunk.size - 1) {
            updateChunk(voxelCoordinate, {x: 0, y: 0, z: 1});
        }

        rollOverMesh.visible = false;

        return true;
    }

    function onClickCallback(intersect) {
        var voxelCoordinate = getVoxelCoordinate(intersect, false);

        if (voxelCoordinate) {
            K3D.dispatch(K3D.events.VOXELS_CALLBACK, {coord: voxelCoordinate, object: object});
        }

        return false;
    }

    return {
        onHover: function (intersect, viewMode) {
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
        onClick: function (intersect, viewMode) {
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
        }
    };
};
