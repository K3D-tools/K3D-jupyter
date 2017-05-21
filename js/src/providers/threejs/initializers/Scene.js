'use strict';

var _ = require('lodash');
var Text2d = require('./../objects/Text2d');
var Config = require('./../../../core/lib/Config');

var planes = {};
var labelsOnEdges = {};
var startingSceneBoundingBox = new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));

function pow10ceil(x) {
    return Math.pow(10, Math.ceil(Math.log10(x)));
}

function generateEdgesPoints(box) {
    return {
        '-x+z': [
            new THREE.Vector3(box.min.x, box.min.y, box.max.z),
            new THREE.Vector3(box.min.x, box.max.y, box.max.z)
        ],
        '+y+z': [
            new THREE.Vector3(box.min.x, box.max.y, box.max.z),
            box.max
        ],
        '+x+z': [
            new THREE.Vector3(box.max.x, box.min.y, box.max.z),
            box.max
        ],
        '-y+z': [
            new THREE.Vector3(box.min.x, box.min.y, box.max.z),
            new THREE.Vector3(box.max.x, box.min.y, box.max.z)
        ],
        '-x-z': [
            box.min,
            new THREE.Vector3(box.min.x, box.max.y, box.min.z)
        ],
        '+y-z': [
            new THREE.Vector3(box.min.x, box.max.y, box.min.z),
            new THREE.Vector3(box.max.x, box.max.y, box.min.z)
        ],
        '+x-z': [
            new THREE.Vector3(box.max.x, box.min.y, box.min.z),
            new THREE.Vector3(box.max.x, box.max.y, box.min.z)
        ],
        '-y-z': [
            box.min,
            new THREE.Vector3(box.max.x, box.min.y, box.min.z)
        ],
        '-x+y': [
            new THREE.Vector3(box.min.x, box.max.y, box.min.z),
            new THREE.Vector3(box.min.x, box.max.y, box.max.z)
        ],
        '-x-y': [
            box.min,
            new THREE.Vector3(box.min.x, box.min.y, box.max.z)
        ],
        '+x+y': [
            new THREE.Vector3(box.max.x, box.max.y, box.min.z),
            box.max
        ],
        '+x-y': [
            new THREE.Vector3(box.max.x, box.min.y, box.min.z),
            new THREE.Vector3(box.max.x, box.min.y, box.max.z)
        ]
    }
}

function rebuildSceneData(K3D) {
    /*jshint validthis:true */
    var promises = [];
    var octree = this.octree = new THREE.Octree({
        //scene: this.scene,
        undeferred: false,
        depthMax: Infinity,
        objectsThreshold: 2,
        overlapPct: 0.5
    });

    this.K3DObjects.children.forEach(function (object) {
        object.traverse(function (object) {
            if (object.geometry &&
                object.geometry.boundingSphere && object.geometry.boundingSphere.radius > 0 &&
                object.interactions) {
                octree.add(object);
            }
        });
    });

    octree.update();

    if (K3D.parameters.gridAutoFit) {

        // Grid generation
        var objectBoundingBox, sceneBoundingBox = startingSceneBoundingBox.clone();

        this.K3DObjects.traverse(function (object) {
            if (object.geometry) {
                objectBoundingBox = object.geometry.boundingBox.clone();
                objectBoundingBox.applyMatrix4(object.matrixWorld);
                sceneBoundingBox.union(objectBoundingBox);
            }
        });

        // cleanup previously data
        if (typeof(planes) !== 'undefined') {
            Object.keys(planes).forEach(function (axis) {
                planes[axis].forEach(function (plane) {
                    this.scene.remove(plane.obj);
                    delete plane.obj;
                }, this);
            }, this);
        }

        if (typeof(labelsOnEdges) !== 'undefined') {
            Object.keys(labelsOnEdges).forEach(function (key) {
                labelsOnEdges[key].labels.forEach(function (label) {
                    label.onRemove();
                }, this);
            }, this);
        }

        // generate new one
        var size = sceneBoundingBox.getSize();
        var majorScale = pow10ceil(Math.max(size.x, size.y, size.z)) / 10.0;
        var minorScale = majorScale / 10.0;
        var unitVectors = {
            'x': new THREE.Vector3(1.0, 0.0, 0.0),
            'y': new THREE.Vector3(0.0, 1.0, 0.0),
            'z': new THREE.Vector3(0.0, 0.0, 1.0)
        };

        sceneBoundingBox.min = new THREE.Vector3(
            Math.floor(sceneBoundingBox.min.x / majorScale) * majorScale,
            Math.floor(sceneBoundingBox.min.y / majorScale) * majorScale,
            Math.floor(sceneBoundingBox.min.z / majorScale) * majorScale);

        sceneBoundingBox.max = new THREE.Vector3(
            Math.ceil(sceneBoundingBox.max.x / majorScale) * majorScale,
            Math.ceil(sceneBoundingBox.max.y / majorScale) * majorScale,
            Math.ceil(sceneBoundingBox.max.z / majorScale) * majorScale);

        size = sceneBoundingBox.getSize();

        planes = {
            'x': [
                {
                    normal: new THREE.Vector3(-1.0, 0.0, 0.0),
                    p1: new THREE.Vector3(sceneBoundingBox.max.x, sceneBoundingBox.min.y, sceneBoundingBox.min.z),
                    p2: sceneBoundingBox.max
                },
                {
                    normal: new THREE.Vector3(1.0, 0.0, 0.0),
                    p1: sceneBoundingBox.min,
                    p2: new THREE.Vector3(sceneBoundingBox.min.x, sceneBoundingBox.max.y, sceneBoundingBox.max.z)
                }],
            'y': [
                {
                    normal: new THREE.Vector3(0.0, -1.0, 0.0),
                    p1: new THREE.Vector3(sceneBoundingBox.min.x, sceneBoundingBox.max.y, sceneBoundingBox.min.z),
                    p2: sceneBoundingBox.max
                },
                {
                    normal: new THREE.Vector3(0.0, 1.0, 0.0),
                    p1: sceneBoundingBox.min,
                    p2: new THREE.Vector3(sceneBoundingBox.max.x, sceneBoundingBox.min.y, sceneBoundingBox.max.z)
                }],
            'z': [
                {
                    normal: new THREE.Vector3(0.0, 0.0, -1.0),
                    p1: new THREE.Vector3(sceneBoundingBox.min.x, sceneBoundingBox.min.y, sceneBoundingBox.max.z),
                    p2: sceneBoundingBox.max
                },
                {
                    normal: new THREE.Vector3(0.0, 0.0, 1.0),
                    p1: sceneBoundingBox.min,
                    p2: new THREE.Vector3(sceneBoundingBox.max.x, sceneBoundingBox.max.y, sceneBoundingBox.min.z)
                }]
        };

        // expand sceneBoundingBox to avoid labels overlapping
        var extendedSceneBoundingBox = new THREE.Box3();

        extendedSceneBoundingBox.min = sceneBoundingBox.min.clone().add(sceneBoundingBox.min.clone()
            .setLength(majorScale * 0.2));
        extendedSceneBoundingBox.max = sceneBoundingBox.max.clone().add(sceneBoundingBox.max.clone()
            .setLength(majorScale * 0.2));

        var originalEdges = generateEdgesPoints(sceneBoundingBox);
        var extendedEdges = generateEdgesPoints(extendedSceneBoundingBox);

        Object.keys(originalEdges).forEach(function (key) {
            labelsOnEdges[key] = {};
            labelsOnEdges[key].v = originalEdges[key];
            labelsOnEdges[key].p = extendedEdges[key];
            labelsOnEdges[key].labels = [];
        });

        // create labels
        Object.keys(labelsOnEdges).forEach(function (key) {
            var iterateAxis = _.difference(['x', 'y', 'z'], key.replace(/[^xyz]/g, '').split(''))[0];
            var iterationCount = size[iterateAxis] / majorScale;
            var deltaValue = unitVectors[iterateAxis].clone().multiplyScalar(majorScale);
            var deltaPosition = unitVectors[iterateAxis].clone().multiplyScalar(
                labelsOnEdges[key].p[0].distanceTo(labelsOnEdges[key].p[1]) / iterationCount
            );

            for (var j = 1; j <= iterationCount - 1; j++) {

                var v = labelsOnEdges[key].v[0].clone().add(deltaValue.clone().multiplyScalar(j));
                var p = labelsOnEdges[key].p[0].clone().add(deltaPosition.clone().multiplyScalar(j));

                var label = new Text2d(new Config({
                    'position': p.toArray(),
                    'referencePoint': 'cc',
                    'color': 0x444444,
                    'text': v[iterateAxis].toString(),
                    'size': 0.75
                }), K3D);

                promises.push(label.then(function (obj) {
                    labelsOnEdges[key].labels.push(obj);
                }));
            }

            // add axis label
            var middleValue = labelsOnEdges[key].v[0].clone().add(
                (new THREE.Vector3()).subVectors(labelsOnEdges[key].v[1], labelsOnEdges[key].v[0]).multiplyScalar(0.5)
            );

            var middlePosition = labelsOnEdges[key].p[0].clone().add(
                (new THREE.Vector3()).subVectors(labelsOnEdges[key].p[1], labelsOnEdges[key].p[0]).multiplyScalar(0.5)
            );

            var middle = middlePosition.add(
                (new THREE.Vector3()).subVectors(middlePosition, middleValue).multiplyScalar(2.0)
            );

            var label = new Text2d(new Config({
                'position': middle.toArray(),
                'referencePoint': 'cc',
                'color': 0x444444,
                'text': iterateAxis,
                'size': 1.0
            }), K3D);

            label.then(function (obj) {
                labelsOnEdges[key].labels.push(obj);
            });
        });

        // create grids
        Object.keys(planes).forEach(function (axis) {
            planes[axis].forEach(function (plane) {
                var vertices = [], colors = [];
                var geometry = new THREE.BufferGeometry();
                var material = new THREE.LineBasicMaterial({vertexColors: THREE.VertexColors});

                var iterableAxes = ['x', 'y', 'z'].filter(function (val) {
                    return val !== axis;
                });

                iterableAxes.forEach(function (iterateAxis) {
                    var delta = unitVectors[iterateAxis].clone().multiplyScalar(minorScale);

                    for (var j = 0; j <= size[iterateAxis] / minorScale; j++) {

                        var p1 = plane.p1.clone().add(delta.clone().multiplyScalar(j));
                        vertices = vertices.concat(p1.toArray());
                        var p2 = plane.p2.clone();
                        p2[iterateAxis] = p1[iterateAxis];
                        vertices = vertices.concat(p2.toArray());

                        if (j % 10 === 0) {
                            colors.push(0.65, 0.65, 0.65);
                            colors.push(0.65, 0.65, 0.65);
                        } else {
                            colors.push(0.85, 0.85, 0.85);
                            colors.push(0.85, 0.85, 0.85);
                        }
                    }
                }, this);

                geometry.addAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
                geometry.addAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
                geometry.computeBoundingSphere();
                geometry.computeBoundingBox();
                plane.obj = new THREE.LineSegments(geometry, material);

                this.scene.add(plane.obj);
            }, this);
        }, this);
    }

    return promises;
}

function refreshGrid() {
    /*jshint validthis:true */
    var visiblePlanes = [];
    Object.keys(planes).forEach(function (axis) {
        var dot1 = planes[axis][0].normal.dot(this.camera.position.clone().sub(planes[axis][0].p1));
        var dot2 = planes[axis][1].normal.dot(this.camera.position.clone().sub(planes[axis][1].p1));

        planes[axis][0].obj.visible = dot1 >= dot2;
        planes[axis][1].obj.visible = dot1 < dot2;

        if (planes[axis][0].obj.visible) {
            visiblePlanes.push('+' + axis);
        }

        if (planes[axis][1].obj.visible) {
            visiblePlanes.push('-' + axis);
        }
    }, this);

    Object.keys(labelsOnEdges).forEach(function (key) {
        var axes = key.match(/.{2}/g);
        var shouldBeVisible = _.intersection(axes, visiblePlanes).length == 1;

        labelsOnEdges[key].labels.forEach(function (label) {
            if (shouldBeVisible) {
                label.show();
            } else {
                label.hide();
            }
        });
    });
}

function raycast(x, y, camera, click, viewMode) {
    /*jshint validthis:true */
    var intersections,
        octreeObjects,
        intersect,
        needRender = false;

    this.raycaster.setFromCamera(new THREE.Vector2(x, y), camera);

    octreeObjects = this.octree.search(
        this.raycaster.ray.origin,
        this.raycaster.ray.far,
        true,
        this.raycaster.ray.direction
    );

    intersections = this.raycaster.intersectOctreeObjects(octreeObjects);

    if (intersections.length > 0) {
        document.body.style.cursor = 'pointer';

        intersections.sort(function (a, b) {
            return a.distance - b.distance;
        });

        intersect = intersections[0];

        if (intersect.object.interactions && intersect.object.interactions.onHover) {
            needRender |= intersect.object.interactions.onHover(intersect, viewMode);
        }

        if (click) {
            if (intersect.object.interactions && intersect.object.interactions.onClick) {
                needRender |= intersect.object.interactions.onClick(intersect, viewMode);
            }
        }
    } else {
        document.body.style.cursor = 'auto';
    }

    return needRender;
}

/**
 * Scene initializer for Three.js library
 * @this K3D.Core~world
 * @method Scene
 * @memberof K3D.Providers.ThreeJS.Initializers
 */
module.exports = {
    startingSceneBoundingBox: startingSceneBoundingBox,

    Init: function (K3D) {
        var lights = [],
            ambientLight = new THREE.AmbientLight(0x111111);

        this.raycaster = new THREE.Raycaster();

        lights[0] = new THREE.PointLight(0xffffff, 1, 0);
        lights[1] = new THREE.PointLight(0xffffff, 1, 0);
        lights[2] = new THREE.PointLight(0xffffff, 1, 0);
        lights[3] = new THREE.PointLight(0xffffff, 1, 0);

        lights[0].position.set(-100, -1000, -500);
        lights[1].position.set(-3000, 1000, -500);

        lights[2].position.set(-5000, 1000, 1000);
        lights[3].position.set(5000, -1000, 1000);

        this.scene = new THREE.Scene();

        this.K3DObjects = new THREE.Group();

        this.scene.add(lights[0]);
        this.scene.add(lights[1]);
        this.scene.add(lights[2]);
        this.scene.add(lights[3]);
        this.scene.add(ambientLight);
        this.scene.add(this.K3DObjects);

        K3D.raycast = raycast.bind(this);
        K3D.rebuildSceneData = rebuildSceneData.bind(this, K3D);
        K3D.refreshGrid = refreshGrid.bind(this);

        K3D.rebuildSceneData();
        K3D.refreshGrid();
    }
};