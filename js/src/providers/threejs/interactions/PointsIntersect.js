const threeMeshBVH = require('three-mesh-bvh');
const THREE = require('three');

// How many world units one pixel covers at the object's distance. The dot shader draws a fixed
// number of pixels, so its radius is a screen length: reading it as a world size picks far too wide.
function pixelToWorld(object, raycaster, K3D) {
    const camera = raycaster.camera;
    const height = (K3D && K3D.getWorld()) ? K3D.getWorld().height : 0;

    if (!camera || !camera.isPerspectiveCamera || !height) {
        return 0.5;
    }

    const centre = new THREE.Vector3().setFromMatrixPosition(object.matrixWorld);
    const distance = camera.position.distanceTo(centre);

    return (2.0 * distance * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2.0)) / height;
}

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

    Intersect(object, K3D) {
        return function (raycaster) {
            const intersects = [];

            const inverseMatrix = new THREE.Matrix4();
            inverseMatrix.copy(object.matrixWorld).invert();

            const ray = raycaster.ray.clone().applyMatrix4(inverseMatrix);
            let closestDistance = Infinity;

            // the dot shader sizes gl_PointSize in pixels, every other one in world units
            const sizeToRadius = object.userData.pointSizeInPixels
                ? pixelToWorld(object, raycaster, K3D) : 0.5;

            let threshold = object.material.size * sizeToRadius || 1;
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
                        if (object.isInstancedMesh) {
                            const matrix = new THREE.Matrix4()
                                .fromArray(object.instanceMatrix.array, point * 16);

                            // the instance scale is relative to the icosahedron, which already has
                            // point_size baked into its radius - and it is already in the object's
                            // own space, so its scale must not divide it a second time
                            localThreshold = (matrix.getMaxScaleOnAxis()
                                * (object.userData.builtPointSize || 1.0)) / 2.0;
                        } else {
                            threshold = object.geometry.attributes.sizes.array[point] * sizeToRadius;
                            localThreshold = threshold
                                / ((object.scale.x + object.scale.y + object.scale.z) / 3);
                        }

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
