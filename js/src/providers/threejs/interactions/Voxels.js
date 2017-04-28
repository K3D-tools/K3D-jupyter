'use strict';

var viewModes = require('./../../../core/lib/viewMode').viewModes;
/**
 * Interactions handlers for Voxels object
 * @method Voxels
 * @memberof K3D.Providers.ThreeJS.Interactions
 */
module.exports = function (object, mesh, rollOverMesh, K3D) {

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

        x = Math.floor(point.x * object.voxelSize.width);
        y = Math.floor(point.y * object.voxelSize.height);
        z = Math.floor(point.z * object.voxelSize.length);

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
                coordinate.x >= mesh.voxel.offsets.x &&
                coordinate.x <= mesh.voxel.offsets.x + mesh.voxel.chunkSize &&
                coordinate.y >= mesh.voxel.offsets.y &&
                coordinate.y <= mesh.voxel.offsets.y + mesh.voxel.chunkSize &&
                coordinate.z >= mesh.voxel.offsets.z &&
                coordinate.z <= mesh.voxel.offsets.z + mesh.voxel.chunkSize) {

                return mesh;
            }
        });
    }

    function rebuildChunk(voxelCoordinate, offset) {
        var nextMesh = findMesh({
            x: voxelCoordinate.x + offset.x,
            y: voxelCoordinate.y + offset.y,
            z: voxelCoordinate.z + offset.z
        });

        if (nextMesh) {
            nextMesh.geometry = nextMesh.voxel.getGeometry(nextMesh.voxel.generate());
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

        object.voxels[i] = K3D.parameters.voxelPaintColor;

        mesh.geometry = mesh.voxel.getGeometry(mesh.voxel.generate());

        // we should handle case when voxelCoordinate is in another chunk
        nextMesh = findMesh(voxelCoordinate);

        if (nextMesh && mesh.uuid !== nextMesh.uuid) {
            nextMesh.geometry = nextMesh.voxel.getGeometry(nextMesh.voxel.generate());
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

        object.voxels[i] = K3D.parameters.voxelPaintColor;

        mesh.geometry = mesh.voxel.getGeometry(mesh.voxel.generate());

        if (voxelCoordinate.x === mesh.voxel.offsets.x) {
            rebuildChunk(voxelCoordinate, {x: -1, y: 0, z: 0});
        }

        if (voxelCoordinate.x === mesh.voxel.offsets.x + mesh.voxel.chunkSize - 1) {
            rebuildChunk(voxelCoordinate, {x: 1, y: 0, z: 0});
        }

        if (voxelCoordinate.y === mesh.voxel.offsets.y) {
            rebuildChunk(voxelCoordinate, {x: 0, y: -1, z: 0});
        }

        if (voxelCoordinate.y === mesh.voxel.offsets.y + mesh.voxel.chunkSize - 1) {
            rebuildChunk(voxelCoordinate, {x: 0, y: 1, z: 0});
        }

        if (voxelCoordinate.z === mesh.voxel.offsets.z) {
            rebuildChunk(voxelCoordinate, {x: 1, y: 0, z: -1});
        }

        if (voxelCoordinate.z === mesh.voxel.offsets.z + mesh.voxel.chunkSize - 1) {
            rebuildChunk(voxelCoordinate, {x: 0, y: 0, z: 1});
        }

        rollOverMesh.visible = false;

        return true;
    }

    K3D.on(K3D.events.VIEW_MODE_CHANGE, function () {
        rollOverMesh.visible = false;
    });

    return {
        onHover: function (intersect, viewMode) {
            switch (viewMode) {
                case viewModes.add:
                    return onHoverAdd(intersect);
                case viewModes.change:
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
                default:
                    return false;
            }
        }
    };
};
