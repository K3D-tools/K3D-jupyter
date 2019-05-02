'use strict';

var THREE = require('three'),
    Text = require('./../objects/Text'),
    MeshLine = require('./../helpers/THREE.MeshLine')(THREE),
    viewModes = require('./../../../core/lib/viewMode').viewModes,
    pow10ceil = require('./../../../core/lib/helpers/math').pow10ceil;

function ensureTwoTicksOnGrids(sceneBoundingBox, majorScale) {
    var size = sceneBoundingBox.getSize(new THREE.Vector3());

    ['x', 'y', 'z'].forEach(function (axis) {
        var dist = size[axis] / majorScale;

        if (dist <= 2.0) {
            sceneBoundingBox.min[axis] -= (1.0 - dist / 2.0 + 0.0001) * majorScale;
            sceneBoundingBox.max[axis] += (1.0 - dist / 2.0 + 0.0001) * majorScale;
        }
    });
}

function getSceneBoundingBox() {
    /*jshint validthis:true */

    var sceneBoundingBox = new THREE.Box3(),
        objectBoundingBox;

    this.K3DObjects.traverse(function (object) {
        var isK3DObject = false,
            ref = object;

        while (ref.parent) {
            if (ref.K3DIdentifier) {
                isK3DObject = true;
                break;
            }

            ref = ref.parent;
        }

        if (isK3DObject &&
            typeof (object.position.z) !== 'undefined' &&
            (object.geometry || object.boundingBox)) {

            if (object.geometry && object.geometry.boundingBox) {
                objectBoundingBox = object.geometry.boundingBox.clone();
            } else {
                objectBoundingBox = object.boundingBox.clone();
            }

            objectBoundingBox.applyMatrix4(object.matrixWorld);
            sceneBoundingBox.union(objectBoundingBox);
        }
    });

    return sceneBoundingBox.isEmpty() ? null : sceneBoundingBox;
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
    };
}

function rebuildSceneData(K3D, grids, force) {
    /*jshint validthis:true */
    var promises = [],
        fullSceneBoundingBox,
        fullSceneDiameter,
        originalEdges,
        extendedEdges,
        size,
        majorScale,
        minorScale,
        sceneBoundingBox = new THREE.Box3().setFromArray(K3D.parameters.grid),
        extendedSceneBoundingBox,
        unitVectors = {
            'x': new THREE.Vector3(1.0, 0.0, 0.0),
            'y': new THREE.Vector3(0.0, 1.0, 0.0),
            'z': new THREE.Vector3(0.0, 0.0, 1.0)
        },
        octree = this.octree = new THREE.Octree({
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

    if (K3D.parameters.gridAutoFit || force) {
        // Grid generation

        if (K3D.parameters.gridAutoFit) {
            sceneBoundingBox = K3D.getSceneBoundingBox() || sceneBoundingBox;
        }

        // cleanup previous data
        Object.keys(grids.planes).forEach(function (axis) {
            grids.planes[axis].forEach(function (plane) {
                this.gridScene.remove(plane.obj);
                delete plane.obj;
            }, this);
        }, this);

        Object.keys(grids.labelsOnEdges).forEach(function (key) {
            grids.labelsOnEdges[key].labels.forEach(function (label) {
                label.onRemove();
            }, this);
        }, this);

        // generate new one
        size = sceneBoundingBox.getSize(new THREE.Vector3());
        majorScale = pow10ceil(Math.max(size.x, size.y, size.z)) / 10.0;
        minorScale = majorScale / 10.0;

        ensureTwoTicksOnGrids(sceneBoundingBox, majorScale);

        sceneBoundingBox.min = new THREE.Vector3(
            Math.floor(sceneBoundingBox.min.x / majorScale) * majorScale,
            Math.floor(sceneBoundingBox.min.y / majorScale) * majorScale,
            Math.floor(sceneBoundingBox.min.z / majorScale) * majorScale);

        sceneBoundingBox.max = new THREE.Vector3(
            Math.ceil(sceneBoundingBox.max.x / majorScale) * majorScale,
            Math.ceil(sceneBoundingBox.max.y / majorScale) * majorScale,
            Math.ceil(sceneBoundingBox.max.z / majorScale) * majorScale);

        size = sceneBoundingBox.getSize(new THREE.Vector3());

        grids.planes = {
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
        extendedSceneBoundingBox = sceneBoundingBox.clone().expandByScalar(majorScale * 0.15);

        originalEdges = generateEdgesPoints(sceneBoundingBox);
        extendedEdges = generateEdgesPoints(extendedSceneBoundingBox);

        Object.keys(originalEdges).forEach(function (key) {
            grids.labelsOnEdges[key] = {};
            grids.labelsOnEdges[key].v = originalEdges[key];
            grids.labelsOnEdges[key].p = extendedEdges[key];
            grids.labelsOnEdges[key].labels = [];
        });

        // create labels
        Object.keys(grids.labelsOnEdges).forEach(function (key) {
            var iterateAxis = _.difference(['x', 'y', 'z'], key.replace(/[^xyz]/g, '').split(''))[0],
                iterationCount = size[iterateAxis] / majorScale,
                deltaValue = unitVectors[iterateAxis].clone().multiplyScalar(majorScale),
                deltaPosition = unitVectors[iterateAxis].clone().multiplyScalar(
                    grids.labelsOnEdges[key].p[0].distanceTo(grids.labelsOnEdges[key].p[1]) / iterationCount
                ),
                j, v, p,
                label,
                middleValue, middlePosition, middle, axisLabel;

            for (j = 1; j <= iterationCount - 1; j++) {

                v = grids.labelsOnEdges[key].v[0].clone().add(deltaValue.clone().multiplyScalar(j));
                p = grids.labelsOnEdges[key].p[0].clone().add(deltaPosition.clone().multiplyScalar(j));


                label = new Text.create({
                    'position': p.toArray(),
                    'reference_point': 'cc',
                    'color': 0x444444,
                    'text': parseFloat((v[iterateAxis]).toFixed(15)).toString(),
                    'size': 0.75
                }, K3D);

                /*jshint loopfunc: true */
                promises.push(label.then(function (obj) {
                    grids.labelsOnEdges[key].labels.push(obj);
                }));
                /*jshint loopfunc: false */
            }

            // add axis label
            middleValue = grids.labelsOnEdges[key].v[0].clone().add(
                (new THREE.Vector3()).subVectors(grids.labelsOnEdges[key].v[1], grids.labelsOnEdges[key].v[0])
                    .multiplyScalar(0.5)
            );

            middlePosition = grids.labelsOnEdges[key].p[0].clone().add(
                (new THREE.Vector3()).subVectors(grids.labelsOnEdges[key].p[1], grids.labelsOnEdges[key].p[0])
                    .multiplyScalar(0.5)
            );

            middle = middlePosition.add(
                (new THREE.Vector3()).subVectors(middlePosition, middleValue).multiplyScalar(2.0)
            );

            axisLabel = new Text.create({
                'position': middle.toArray(),
                'reference_point': 'cc',
                'color': 0x444444,
                'text': K3D.parameters.axes[['x', 'y', 'z'].indexOf(iterateAxis)],
                'size': 1.0
            }, K3D);

            axisLabel.then(function (obj) {
                grids.labelsOnEdges[key].labels.push(obj);
            });
        });

        // create grids
        Object.keys(grids.planes).forEach(function (axis) {
            grids.planes[axis].forEach(function (plane) {
                var vertices = [], widths = [], colors = [],
                    iterableAxes = ['x', 'y', 'z'].filter(function (val) {
                        return val !== axis;
                    }),
                    line = new MeshLine.MeshLine(),
                    material = new MeshLine.MeshLineMaterial({
                        color: new THREE.Color(1.0, 1.0, 1.0),
                        opacity: 0.75,
                        sizeAttenuation: true,
                        transparent: true,
                        lineWidth: minorScale * 0.05,
                        resolution: new THREE.Vector2(K3D.getWorld().width, K3D.getWorld().height),
                        side: THREE.DoubleSide
                    });

                iterableAxes.forEach(function (iterateAxis) {
                    var delta = unitVectors[iterateAxis].clone().multiplyScalar(minorScale),
                        p1, p2, j;

                    for (j = 0; j <= size[iterateAxis] / minorScale; j++) {
                        p1 = plane.p1.clone().add(delta.clone().multiplyScalar(j));
                        vertices = vertices.concat(p1.toArray());
                        p2 = plane.p2.clone();
                        p2[iterateAxis] = p1[iterateAxis];
                        vertices = vertices.concat(p2.toArray());

                        if (j % 10 === 0) {
                            widths.push(1.5, 1.5);
                            colors.push(0.65, 0.65, 0.65);
                            colors.push(0.65, 0.65, 0.65);
                        } else {
                            widths.push(1.0, 1.0);
                            colors.push(0.9, 0.9, 0.9);
                            colors.push(0.9, 0.9, 0.9);
                        }
                    }
                }, this);

                line.setGeometry(new Float32Array(vertices), true, widths, colors);
                line.geometry.computeBoundingSphere();
                line.geometry.computeBoundingBox();

                plane.obj = new THREE.Mesh(line.geometry, material);

                this.gridScene.add(plane.obj);
            }, this);
        }, this);
    }

    // Dynamic setting far clipping plane
    fullSceneBoundingBox = sceneBoundingBox.clone();
    Object.keys(grids.planes).forEach(function (axis) {
        grids.planes[axis].forEach(function (plane) {
            fullSceneBoundingBox.union(plane.obj.geometry.boundingBox.clone());
        }, this);
    }, this);

    fullSceneDiameter = fullSceneBoundingBox.getSize(new THREE.Vector3()).length();

    this.camera.far = fullSceneDiameter * 10;
    this.camera.near = fullSceneDiameter * 0.0001;
    this.camera.updateProjectionMatrix();

    return promises;
}

function refreshGrid(K3D, grids) {
    /*jshint validthis:true */
    var visiblePlanes = [],
        cameraDirection = new THREE.Vector3();

    this.camera.getWorldDirection(cameraDirection);

    Object.keys(grids.planes).forEach(function (axis) {
        var dot1 = grids.planes[axis][0].normal.dot(cameraDirection),
            dot2 = grids.planes[axis][1].normal.dot(cameraDirection);

        grids.planes[axis][0].obj.visible = dot1 <= dot2 && K3D.parameters.gridVisible;
        grids.planes[axis][1].obj.visible = dot1 > dot2 && K3D.parameters.gridVisible;

        if (grids.planes[axis][0].obj.visible) {
            visiblePlanes.push('+' + axis);
        }

        if (grids.planes[axis][1].obj.visible) {
            visiblePlanes.push('-' + axis);
        }
    }, this);

    Object.keys(grids.labelsOnEdges).forEach(function (key) {
        var axes = key.match(/.{2}/g),
            shouldBeVisible = _.intersection(axes, visiblePlanes).length === 1;

        grids.labelsOnEdges[key].labels.forEach(function (label) {
            if (shouldBeVisible && K3D.parameters.gridVisible) {
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
    Init: function (K3D) {
        var initialLightIntensity = {
                ambient: 0.2,
                key: 0.4,
                head: 0.15,
                fill: 0.15,
                back: 0.1
            },
            ambientLight = new THREE.AmbientLight(0xffffff),
            grids = {
                planes: {},
                labelsOnEdges: {}
            },
            self = this;

        this.lights = [];
        this.raycaster = new THREE.Raycaster();

        // https://www.vtk.org/doc/release/5.0/html/a01682.html
        // A LightKit consists of three lights, a key light, a fill light, and a headlight. The main light is the key
        // light. It is usually positioned so that it appears like an overhead light (like the sun, or a ceiling light).
        // It is generally positioned to shine down on the scene from about a 45 degree angle vertically and at least a
        // little offset side to side. The key light usually at least about twice as bright as the total of all other
        // lights in the scene to provide good modeling of object features.

        // The other lights in the kit (the fill light, headlight, and a pair of back lights) are weaker sources that
        // provide extra illumination to fill in the spots that the key light misses. The fill light is usually
        // positioned across from or opposite from the key light (though still on the same side of the object as the
        // camera) in order to simulate diffuse reflections from other objects in the scene. The headlight, always
        // located at the position of the camera, reduces the contrast between areas lit by the key and fill light.
        // The two back lights, one on the left of the object as seen from the observer and one on the right, fill on
        // the high-contrast areas behind the object. To enforce the relationship between the different lights, the
        // intensity of the fill, back and headlights are set as a ratio to the key light brightness. Thus, the
        // brightness of all the lights in the scene can be changed by changing the key light intensity.

        this.keyLight = new THREE.DirectionalLight(0xffffff);  //key
        this.headLight = new THREE.DirectionalLight(0xffffff); //head
        this.fillLight = new THREE.DirectionalLight(0xffffff); //fill
        this.backLight = new THREE.DirectionalLight(0xffffff); //back

        this.keyLight.position.set(0.25, 1, 1.0);
        this.headLight.position.set(0, 0, 1);
        this.fillLight.position.set(-0.25, -1, 1.0);
        this.backLight.position.set(-2.5, 0.4, -1);

        this.scene = new THREE.Scene();
        this.gridScene = new THREE.Scene();

        this.K3DObjects = new THREE.Group();

        [this.keyLight, this.headLight, this.fillLight, this.backLight].forEach(function (light) {
            self.camera.add(light);
            self.camera.add(light.target);
            // self.scene.add(new THREE.DirectionalLightHelper(light, 1.0, 0xff0000));
        });

        this.scene.add(ambientLight);
        this.scene.add(this.camera);
        this.scene.add(this.K3DObjects);

        K3D.rebuildSceneData = rebuildSceneData.bind(this, K3D, grids);
        K3D.getSceneBoundingBox = getSceneBoundingBox.bind(this);
        K3D.refreshGrid = refreshGrid.bind(this, K3D, grids);

        Promise.all(K3D.rebuildSceneData()).then(function () {
            K3D.refreshGrid();
            K3D.render();
        });

        this.recalculateLights = function (value) {
            if (value <= 1.0) {
                ambientLight.intensity = 1.0 - (1.0 - initialLightIntensity.ambient) * value;
            } else {
                ambientLight.intensity = initialLightIntensity.ambient;
            }

            self.keyLight.intensity = initialLightIntensity.key * value;
            self.headLight.intensity = initialLightIntensity.head * value;
            self.fillLight.intensity = initialLightIntensity.fill * value;
            self.backLight.intensity = initialLightIntensity.back * value;

            self.backLight.visible = self.headLight.visible = self.fillLight.visible = self.backLight.visible =
                value > 0.0;
        };

        K3D.on(K3D.events.MOUSE_MOVE, function (coord) {
            if (K3D.parameters.viewMode !== viewModes.view) {
                if (raycast.call(self, coord.x, coord.y, self.camera, false, K3D.parameters.viewMode) &&
                    !K3D.autoRendering) {
                    K3D.render();
                }
            }
        });

        K3D.on(K3D.events.MOUSE_CLICK, function (coord) {
            if (K3D.parameters.viewMode !== viewModes.view) {
                if (raycast.call(self, coord.x, coord.y, self.camera, true, K3D.parameters.viewMode) &&
                    !K3D.autoRendering) {
                    K3D.render();
                }
            }
        });

        K3D.on(K3D.events.RESIZED, function () {
            // update outlines
            Object.keys(grids.planes).forEach(function (axis) {
                grids.planes[axis].forEach(function (plane) {
                    var objResolution = plane.obj.material.uniforms.resolution;

                    objResolution.value.x = K3D.getWorld().width;
                    objResolution.value.y = K3D.getWorld().height;
                }, this);
            }, this);
        });
    }
};
