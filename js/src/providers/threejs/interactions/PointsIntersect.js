const threeMeshBVH = require('three-mesh-bvh');
const THREE = require('three');

module.exports = {
    prepareGeometry(geometry) {
        const bvhGeometry = geometry.clone();
        const indices = [];

        let verticesLength = bvhGeometry.attributes.position.count;
        for (let i = 0, l = verticesLength; i < l; i++) {
            indices.push(i, i, i);
        }
        bvhGeometry.setIndex(indices);

        return bvhGeometry;
    },

    Intersect(object) {
        return function (raycaster) {
            let intersects = [];

            const inverseMatrix = new THREE.Matrix4();
            inverseMatrix.copy(object.matrixWorld).invert();
            raycaster.ray.applyMatrix4(inverseMatrix);

            const { ray } = raycaster;
            let closestDistance = Infinity;

            let threshold = object.material.size / 2.0 || 1;
            let localThreshold = threshold / ((object.scale.x + object.scale.y + object.scale.z) / 3);
            let localThresholdSq = localThreshold * localThreshold;

            let ret = null;

            object.geometry.boundsTree.shapecast({
                boundsTraverseOrder: function (box) {
                    return box.distanceToPoint(ray.origin);
                },
                intersectsBounds: function (box, isLeaf, score) {
                    if (score > closestDistance) {
                        return threeMeshBVH.NOT_INTERSECTED;
                    }

                    box.expandByScalar(localThreshold);
                    return ray.intersectsBox(box) ? threeMeshBVH.INTERSECTED : threeMeshBVH.NOT_INTERSECTED;

                },
                intersectsTriangle: function (triangle, triangleIndex) {
                    const distancesToRaySq = ray.distanceSqToPoint(triangle.a);

                    if (object.geometry.attributes.sizes || object.isInstancedMesh) {

                        if (object.geometry.attributes.sizes) {
                            threshold = object.geometry.attributes.sizes.array[triangleIndex] / 2.0;
                        }

                        if (object.isInstancedMesh) {
                            let matrix = new THREE.Matrix4().fromArray(object.instanceMatrix.array, triangleIndex * 16);
                            threshold = matrix.getMaxScaleOnAxis() / 2.0;
                        }

                        localThreshold = threshold / ((object.scale.x + object.scale.y + object.scale.z) / 3);
                        localThresholdSq = localThreshold * localThreshold;
                    }

                    if (distancesToRaySq < localThresholdSq) {
                        const distanceToPoint = ray.origin.distanceTo(triangle.a);

                        if (distanceToPoint < closestDistance) {
                            closestDistance = distanceToPoint;

                            ret = {
                                object: object,
                                point: triangle.a,
                                distance: distanceToPoint,
                                index: triangleIndex
                            }
                        }
                    }
                }
            });

            if (closestDistance !== Infinity) {
                intersects.push(ret);
            }

            return intersects;
        }
    },
};
