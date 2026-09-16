const threeMeshBVH = require('three-mesh-bvh');
const THREE = require('three');

module.exports = {
    prepareGeometry(geometry) {
        const bvhGeometry = geometry.clone();
        const indices = [];

        const verticesLength = bvhGeometry.attributes.position.count;
        for (let i = 0, l = verticesLength; i < l; i++) {
            indices.push(i, i, i);
        }
        bvhGeometry.setIndex(indices);

        return bvhGeometry;
    },

    Intersect(object) {
        return function (raycaster) {
            const intersects = [];

            const inverseMatrix = new THREE.Matrix4();
            inverseMatrix.copy(object.matrixWorld).invert();

            const ray = raycaster.ray.clone().applyMatrix4(inverseMatrix);
            let closestDistance = Infinity;

            let threshold = object.material.size / 2.0 || 1;
            let localThreshold = threshold / ((object.scale.x + object.scale.y + object.scale.z) / 3);
            let localThresholdSq = localThreshold * localThreshold;

            let ret = null;

            // the BVH sorted the index buffer it was built on, so the slot it reports is not
            // the point number; the index it kept still says which point each slot holds
            const bvhIndex = object.geometry.boundsTree.geometry.index;

            object.geometry.boundsTree.shapecast({
                boundsTraverseOrder(box) {
                    return box.distanceToPoint(ray.origin);
                },
                intersectsBounds(box, isLeaf, score) {
                    if (score > closestDistance) {
                        return threeMeshBVH.NOT_INTERSECTED;
                    }

                    box.expandByScalar(localThreshold);
                    return ray.intersectsBox(box) ? threeMeshBVH.INTERSECTED : threeMeshBVH.NOT_INTERSECTED;
                },
                intersectsTriangle(triangle, triangleIndex) {
                    const point = bvhIndex.getX(triangleIndex * 3);
                    const distancesToRaySq = ray.distanceSqToPoint(triangle.a);

                    if (object.geometry.attributes.sizes || object.isInstancedMesh) {
                        if (object.geometry.attributes.sizes) {
                            threshold = object.geometry.attributes.sizes.array[point] / 2.0;
                        }

                        if (object.isInstancedMesh) {
                            const matrix = new THREE.Matrix4()
                                .fromArray(object.instanceMatrix.array, point * 16);

                            // the instance scale is relative to the icosahedron, which already
                            // has point_size baked into its radius
                            threshold = (matrix.getMaxScaleOnAxis()
                                * (object.userData.builtPointSize || 1.0)) / 2.0;
                        }

                        localThreshold = threshold / ((object.scale.x + object.scale.y + object.scale.z) / 3);
                        localThresholdSq = localThreshold * localThreshold;
                    }

                    if (distancesToRaySq < localThresholdSq) {
                        const distanceToPoint = ray.origin.distanceTo(triangle.a);

                        if (distanceToPoint < closestDistance) {
                            closestDistance = distanceToPoint;

                            const worldPoint = triangle.a.clone().applyMatrix4(object.matrixWorld);

                            ret = {
                                object,
                                point: worldPoint,
                                distance: raycaster.ray.origin.distanceTo(worldPoint),
                                index: point,
                            };
                        }
                    }
                },
            });

            if (closestDistance !== Infinity) {
                intersects.push(ret);
            }

            return intersects;
        };
    },
};
