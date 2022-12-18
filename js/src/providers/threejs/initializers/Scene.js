const THREE = require('three');
const _ = require('../../../lodash');
const Text = require('../objects/Text');
const Vectors = require('../objects/Vectors');
const MeshLine = require('../helpers/THREE.MeshLine')(THREE);
const { viewModes } = require('../../../core/lib/viewMode');
const { pow10ceil } = require('../../../core/lib/helpers/math');

let rebuildSceneDataPromises = null;

function generateAxesHelper(K3D, axesHelper) {
    const promises = [];
    const colors = K3D.parameters.axesHelperColors;
    const directions = {
        x: [1, 0, 0],
        y: [0, 1, 0],
        z: [0, 0, 1],
    };
    const order = {
        x: 0,
        y: 1,
        z: 2,
    };
    const labelColor = new THREE.Color(K3D.parameters.labelColor);

    ['x', 'y', 'z'].forEach((axis, i) => {
        const label = Text.create({
            position: new THREE.Vector3().fromArray(directions[axis]).multiplyScalar(1.1).toArray(),
            reference_point: 'cc',
            color: labelColor,
            text: K3D.parameters.axes[order[axis]],
            size: 0.75,
        }, K3D, axesHelper);

        promises.push(label.then((obj) => {
            axesHelper[axis] = obj;
            axesHelper[axis].color = colors[i];
        }));
    });

    const arrows = Vectors.create({
        colors: { data: [colors[0], colors[0], colors[1], colors[1], colors[2], colors[2]] },
        origins: { data: [0, 0, 0, 0, 0, 0, 0, 0, 0] },
        vectors: { data: [].concat(directions.x, directions.y, directions.z) },
        line_width: 0.05,
        head_size: 2.5,
    }, K3D);

    promises.push(arrows.then((obj) => {
        axesHelper.arrows = obj;
        axesHelper.scene.add(obj);
    }));

    return promises;
}

function getSceneBoundingBox(K3D) {
    /* jshint validthis:true */

    const sceneBoundingBox = new THREE.Box3();
    let objectBoundingBox;
    let world = K3D.getWorld();

    Object.keys(world.ObjectsListJson).forEach(function (K3DIdentifier) {
        let k3dObject = world.ObjectsById[K3DIdentifier];

        if(!k3dObject) {
            return
        }

        k3dObject.traverse((object) => {
            if (object && typeof (object.position.z) !== 'undefined'
                && object.visible
                && (object.geometry || object.boundingBox)) {
                if (object.geometry && object.geometry.boundingBox) {
                    objectBoundingBox = object.geometry.boundingBox.clone();
                } else if (object.boundingBox) {
                    objectBoundingBox = object.boundingBox.clone();
                } else {
                    console.log('Object without bbox');
                    return;
                }

                objectBoundingBox.applyMatrix4(object.matrixWorld);
                sceneBoundingBox.union(objectBoundingBox);
            }
        });
    });

    // one point on scene?
    if (sceneBoundingBox.getSize(new THREE.Vector3()).lengthSq() < Number.EPSILON) {
        sceneBoundingBox.max.addScalar(0.1);
    }

    return sceneBoundingBox.isEmpty() ? null : sceneBoundingBox;
}

function generateEdgesPoints(box) {
    return {
        '-x+z': [
            new THREE.Vector3(box.min.x, box.min.y, box.max.z),
            new THREE.Vector3(box.min.x, box.max.y, box.max.z),
        ],
        '+y+z': [
            new THREE.Vector3(box.min.x, box.max.y, box.max.z),
            box.max,
        ],
        '+x+z': [
            new THREE.Vector3(box.max.x, box.min.y, box.max.z),
            box.max,
        ],
        '-y+z': [
            new THREE.Vector3(box.min.x, box.min.y, box.max.z),
            new THREE.Vector3(box.max.x, box.min.y, box.max.z),
        ],
        '-x-z': [
            box.min,
            new THREE.Vector3(box.min.x, box.max.y, box.min.z),
        ],
        '+y-z': [
            new THREE.Vector3(box.min.x, box.max.y, box.min.z),
            new THREE.Vector3(box.max.x, box.max.y, box.min.z),
        ],
        '+x-z': [
            new THREE.Vector3(box.max.x, box.min.y, box.min.z),
            new THREE.Vector3(box.max.x, box.max.y, box.min.z),
        ],
        '-y-z': [
            box.min,
            new THREE.Vector3(box.max.x, box.min.y, box.min.z),
        ],
        '-x+y': [
            new THREE.Vector3(box.min.x, box.max.y, box.min.z),
            new THREE.Vector3(box.min.x, box.max.y, box.max.z),
        ],
        '-x-y': [
            box.min,
            new THREE.Vector3(box.min.x, box.min.y, box.max.z),
        ],
        '+x+y': [
            new THREE.Vector3(box.max.x, box.max.y, box.min.z),
            box.max,
        ],
        '+x-y': [
            new THREE.Vector3(box.max.x, box.min.y, box.min.z),
            new THREE.Vector3(box.max.x, box.min.y, box.max.z),
        ],
    };
}

function cleanup(grids, gridScene) {
    Object.keys(grids.planes).forEach((axis) => {
        grids.planes[axis].forEach((plane) => {
            gridScene.remove(plane.obj);
            delete plane.obj;
        });
    });

    Object.keys(grids.labelsOnEdges).forEach((key) => {
        grids.labelsOnEdges[key].labels.forEach((label) => {
            label.onRemove();
        });
    });
}

function rebuildSceneData(K3D, grids, axesHelper, force) {
    /* jshint validthis:true, maxstatements:false */
    const that = this;

    if (rebuildSceneDataPromises) {
        return Promise.all(rebuildSceneDataPromises).then(() => rebuildSceneData.bind(that)(
            K3D,
            grids,
            axesHelper,
            force,
        ));
    }

    const promises = [];
    let originalEdges;
    let updateAxesHelper;
    let extendedEdges;
    let size;
    let majorScale;
    let minorScale;
    let sceneBoundingBox = new THREE.Box3().setFromArray(K3D.parameters.grid);
    let extendedSceneBoundingBox;
    const unitVectors = {
        x: new THREE.Vector3(1.0, 0.0, 0.0),
        y: new THREE.Vector3(0.0, 1.0, 0.0),
        z: new THREE.Vector3(0.0, 0.0, 1.0),
    };
    const gridColor = new THREE.Color(K3D.parameters.gridColor);
    const labelColor = new THREE.Color(K3D.parameters.labelColor);

    // axes Helper
    updateAxesHelper = !K3D.parameters.axesHelper || (K3D.parameters.axesHelper && !axesHelper.x);

    if (axesHelper.x && !updateAxesHelper) {
        // has axes labels changed?
        updateAxesHelper |= K3D.parameters.axes[0] !== axesHelper.x.text
            || K3D.parameters.axes[1] !== axesHelper.y.text
            || K3D.parameters.axes[2] !== axesHelper.z.text;

        // has axes colors changed?
        updateAxesHelper |= K3D.parameters.axesHelperColors[0] !== axesHelper.x.color
            || K3D.parameters.axesHelperColors[1] !== axesHelper.y.color
            || K3D.parameters.axesHelperColors[2] !== axesHelper.z.color;
    }

    if (updateAxesHelper) {
        ['x', 'y', 'z'].forEach((axis) => {
            if (axesHelper[axis]) {
                axesHelper[axis].onRemove();
                axesHelper.scene.remove(axesHelper[axis]);
                axesHelper[axis] = null;
            }
        });

        if (axesHelper.arrows) {
            axesHelper.scene.remove(axesHelper.arrows);
            axesHelper.arrows = null;
        }
    }

    if (K3D.parameters.axesHelper > 1) {
        axesHelper.width = K3D.parameters.axesHelper;
        axesHelper.height = K3D.parameters.axesHelper;
    } else if (K3D.parameters.axesHelper > 0) {
        axesHelper.width = 100;
        axesHelper.height = 100;
    }

    if (updateAxesHelper) {
        if (K3D.parameters.axesHelper > 0) {
            generateAxesHelper(K3D, axesHelper).forEach((p) => {
                promises.push(p);
            });
        }
    }

    if (K3D.parameters.gridAutoFit || force) {
        // Grid generation

        sceneBoundingBox = K3D.getSceneBoundingBox() || sceneBoundingBox;

        // cleanup previous data
        cleanup(grids, this.gridScene);

        // generate new one
        size = sceneBoundingBox.getSize(new THREE.Vector3());
        majorScale = pow10ceil(Math.max(size.x, size.y, size.z)) / 10.0;
        minorScale = majorScale / 10.0;

        ['x', 'y', 'z'].forEach((axis) => {
            if (sceneBoundingBox.min[axis] === sceneBoundingBox.max[axis]) {
                sceneBoundingBox.min[axis] -= majorScale / 2.0;
                sceneBoundingBox.max[axis] += majorScale / 2.0;
            }
        });
        size = sceneBoundingBox.getSize(new THREE.Vector3());

        sceneBoundingBox.min = new THREE.Vector3(
            Math.floor(sceneBoundingBox.min.x / majorScale) * majorScale,
            Math.floor(sceneBoundingBox.min.y / majorScale) * majorScale,
            Math.floor(sceneBoundingBox.min.z / majorScale) * majorScale,
        );

        sceneBoundingBox.max = new THREE.Vector3(
            Math.ceil(sceneBoundingBox.max.x / majorScale) * majorScale,
            Math.ceil(sceneBoundingBox.max.y / majorScale) * majorScale,
            Math.ceil(sceneBoundingBox.max.z / majorScale) * majorScale,
        );

        size = sceneBoundingBox.getSize(new THREE.Vector3());

        grids.planes = {
            x: [
                {
                    normal: new THREE.Vector3(-1.0, 0.0, 0.0),
                    p1: new THREE.Vector3(sceneBoundingBox.max.x, sceneBoundingBox.min.y, sceneBoundingBox.min.z),
                    p2: sceneBoundingBox.max,
                },
                {
                    normal: new THREE.Vector3(1.0, 0.0, 0.0),
                    p1: sceneBoundingBox.min,
                    p2: new THREE.Vector3(sceneBoundingBox.min.x, sceneBoundingBox.max.y, sceneBoundingBox.max.z),
                }],
            y: [
                {
                    normal: new THREE.Vector3(0.0, -1.0, 0.0),
                    p1: new THREE.Vector3(sceneBoundingBox.min.x, sceneBoundingBox.max.y, sceneBoundingBox.min.z),
                    p2: sceneBoundingBox.max,
                },
                {
                    normal: new THREE.Vector3(0.0, 1.0, 0.0),
                    p1: sceneBoundingBox.min,
                    p2: new THREE.Vector3(sceneBoundingBox.max.x, sceneBoundingBox.min.y, sceneBoundingBox.max.z),
                }],
            z: [
                {
                    normal: new THREE.Vector3(0.0, 0.0, -1.0),
                    p1: new THREE.Vector3(sceneBoundingBox.min.x, sceneBoundingBox.min.y, sceneBoundingBox.max.z),
                    p2: sceneBoundingBox.max,
                },
                {
                    normal: new THREE.Vector3(0.0, 0.0, 1.0),
                    p1: sceneBoundingBox.min,
                    p2: new THREE.Vector3(sceneBoundingBox.max.x, sceneBoundingBox.max.y, sceneBoundingBox.min.z),
                }],
        };

        // expand sceneBoundingBox to avoid labels overlapping
        extendedSceneBoundingBox = sceneBoundingBox.clone().expandByScalar(majorScale * 0.15);

        originalEdges = generateEdgesPoints(sceneBoundingBox);
        extendedEdges = generateEdgesPoints(extendedSceneBoundingBox);

        Object.keys(originalEdges).forEach((key) => {
            grids.labelsOnEdges[key] = {};
            grids.labelsOnEdges[key].v = originalEdges[key];
            grids.labelsOnEdges[key].p = extendedEdges[key];
            grids.labelsOnEdges[key].labels = [];
        });

        // create labels
        Object.keys(grids.labelsOnEdges).forEach((key) => {
            const iterateAxis = _.difference(['x', 'y', 'z'], key.replace(/[^xyz]/g, '').split(''))[0];
            let iterationCount = size[iterateAxis] / majorScale;

            let deltaValue = unitVectors[iterateAxis].clone().multiplyScalar(majorScale);
            let deltaPosition = unitVectors[iterateAxis].clone().multiplyScalar(
                grids.labelsOnEdges[key].p[0].distanceTo(grids.labelsOnEdges[key].p[1]) / iterationCount,
            );
            let j;
            let v;
            let p;
            let label;

            if (iterationCount <= 2) {
                const originalIterationCount = iterationCount;

                iterationCount = originalIterationCount * 5;
                deltaValue = unitVectors[iterateAxis].clone()
                    .multiplyScalar((originalIterationCount * majorScale) / iterationCount);
                deltaPosition = unitVectors[iterateAxis].clone().multiplyScalar(
                    grids.labelsOnEdges[key].p[0].distanceTo(grids.labelsOnEdges[key].p[1]) / iterationCount,
                );
            }

            for (j = 1; j <= iterationCount - 1; j++) {
                v = grids.labelsOnEdges[key].v[0].clone().add(deltaValue.clone().multiplyScalar(j));
                p = grids.labelsOnEdges[key].p[0].clone().add(deltaPosition.clone().multiplyScalar(j));

                label = Text.create({
                    position: p.toArray(),
                    reference_point: 'cc',
                    color: labelColor,
                    text: parseFloat((v[iterateAxis]).toFixed(15)).toString(),
                    size: 0.75,
                }, K3D);

                /* jshint loopfunc: true */
                promises.push(label.then((obj) => {
                    grids.labelsOnEdges[key].labels.push(obj);
                }));
                /* jshint loopfunc: false */
            }

            // add axis label
            const middleValue = grids.labelsOnEdges[key].v[0].clone().add(
                (new THREE.Vector3()).subVectors(grids.labelsOnEdges[key].v[1], grids.labelsOnEdges[key].v[0])
                    .multiplyScalar(0.5),
            );

            const middlePosition = grids.labelsOnEdges[key].p[0].clone().add(
                (new THREE.Vector3()).subVectors(grids.labelsOnEdges[key].p[1], grids.labelsOnEdges[key].p[0])
                    .multiplyScalar(0.5),
            );

            const middle = middlePosition.add(
                (new THREE.Vector3()).subVectors(middlePosition, middleValue).multiplyScalar(2.0),
            );

            const axisLabel = Text.create({
                position: middle.toArray(),
                reference_point: 'cc',
                color: labelColor,
                text: K3D.parameters.axes[['x', 'y', 'z'].indexOf(iterateAxis)],
                size: 1.0,
            }, K3D);

            axisLabel.then((obj) => {
                grids.labelsOnEdges[key].labels.push(obj);
            });
        });

        // create grids
        Object.keys(grids.planes).forEach(function (axis) {
            grids.planes[axis].forEach(function (plane) {
                let vertices = [];
                const widths = [];
                const colors = [];
                const iterableAxes = ['x', 'y', 'z'].filter((val) => val !== axis);
                const line = new MeshLine.MeshLine();
                const material = new MeshLine.MeshLineMaterial({
                    color: new THREE.Color(1.0, 1.0, 1.0),
                    opacity: 0.75,
                    sizeAttenuation: true,
                    transparent: true,
                    lineWidth: minorScale * 0.05,
                    resolution: new THREE.Vector2(K3D.getWorld().width, K3D.getWorld().height),
                    side: THREE.DoubleSide,
                });

                iterableAxes.forEach((iterateAxis) => {
                    const delta = unitVectors[iterateAxis].clone().multiplyScalar(minorScale);
                    let p1;
                    let p2;
                    let j;

                    for (j = 0; j <= size[iterateAxis] / minorScale; j++) {
                        p1 = plane.p1.clone().add(delta.clone().multiplyScalar(j));
                        vertices = vertices.concat(p1.toArray());
                        p2 = plane.p2.clone();
                        p2[iterateAxis] = p1[iterateAxis];
                        vertices = vertices.concat(p2.toArray());

                        if (j % 10 === 0) {
                            widths.push(1.5, 1.5);
                            colors.push(gridColor.r * 0.72, gridColor.g * 0.72, gridColor.b * 0.72);
                            colors.push(gridColor.r * 0.72, gridColor.g * 0.72, gridColor.b * 0.72);
                        } else {
                            widths.push(1.0, 1.0);
                            colors.push(gridColor.r, gridColor.g, gridColor.b);
                            colors.push(gridColor.r, gridColor.g, gridColor.b);
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
    const fullSceneBoundingBox = sceneBoundingBox.clone();
    Object.keys(grids.planes).forEach(function (axis) {
        grids.planes[axis].forEach((plane) => {
            fullSceneBoundingBox.union(plane.obj.geometry.boundingBox.clone());
        }, this);
    }, this);

    const fullSceneDiameter = fullSceneBoundingBox.getSize(new THREE.Vector3()).length();

    const camDistance = (fullSceneDiameter / 2.0) / Math.sin(THREE.Math.degToRad(K3D.parameters.cameraFov / 2.0));

    this.camera.far = (camDistance + fullSceneDiameter / 2) * 5.0;
    this.camera.near = fullSceneDiameter * 0.0001;
    this.camera.updateProjectionMatrix();

    rebuildSceneDataPromises = promises;

    return Promise.all(promises).then((v) => {
        rebuildSceneDataPromises = null;
        return v;
    });
}

function refreshGrid(K3D, grids) {
    /* jshint validthis:true */
    const visiblePlanes = [];
    const cameraDirection = new THREE.Vector3();

    this.camera.getWorldDirection(cameraDirection);

    Object.keys(grids.planes).forEach((axis) => {
        const dot1 = grids.planes[axis][0].normal.dot(cameraDirection);
        const dot2 = grids.planes[axis][1].normal.dot(cameraDirection);

        grids.planes[axis][0].obj.visible = dot1 <= dot2 && K3D.parameters.gridVisible;
        grids.planes[axis][1].obj.visible = dot1 > dot2 && K3D.parameters.gridVisible;

        if (grids.planes[axis][0].obj.visible) {
            visiblePlanes.push(`+${axis}`);
        }

        if (grids.planes[axis][1].obj.visible) {
            visiblePlanes.push(`-${axis}`);
        }
    }, this);

    Object.keys(grids.labelsOnEdges).forEach((key) => {
        const axes = key.match(/.{2}/g);
        const shouldBeVisible = _.intersection(axes, visiblePlanes).length === 1;

        grids.labelsOnEdges[key].labels.forEach((label) => {
            if (shouldBeVisible && K3D.parameters.gridVisible) {
                label.show();
            } else {
                label.hide();
            }
        });
    });
}

function raycast(K3D, x, y, camera, click, viewMode) {
    /* jshint validthis:true */
    const meshes = [];
    let intersect;
    let needRender = false;

    this.raycaster.setFromCamera(new THREE.Vector2(x, y), camera);

    this.K3DObjects.traverse((object) => {
        if (object.interactions) {
            if (object.geometry && object.geometry.attributes.position.count === 0) {
                return;
            }

            meshes.push(object);
        }
    });

    if (meshes.length > 0) {
        intersect = this.raycaster.intersectObjects(meshes);

        if (intersect.length > 0) {
            intersect = intersect[0];
            K3D.getWorld().targetDOMNode.style.cursor = 'pointer';

            if (!click && intersect.object.interactions && intersect.object.interactions.onHover) {
                needRender |= intersect.object.interactions.onHover(intersect, viewMode);
            }

            if (click && intersect.object.interactions && intersect.object.interactions.onClick) {
                needRender |= intersect.object.interactions.onClick(intersect, viewMode);
            }
        } else {
            K3D.getWorld().targetDOMNode.style.cursor = 'auto';
        }
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
    Init(K3D) {
        const initialLightIntensity = {
            ambient: 0.2,
            key: 0.4,
            head: 0.15,
            fill: 0.15,
            back: 0.1,
        };
        const ambientLight = new THREE.AmbientLight(0xffffff);
        const grids = {
            planes: {},
            labelsOnEdges: {},
        };
        const self = this;

        this.lights = [];
        this.raycaster = new THREE.Raycaster();
        this.raycaster.firstHitOnly = true;

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

        this.keyLight = new THREE.DirectionalLight(0xffffff); // key
        this.headLight = new THREE.DirectionalLight(0xffffff); // head
        this.fillLight = new THREE.DirectionalLight(0xffffff); // fill
        this.backLight = new THREE.DirectionalLight(0xffffff); // back

        this.keyLight.position.set(0.25, 1, 1.0);
        this.headLight.position.set(0, 0, 1);
        this.fillLight.position.set(-0.25, -1, 1.0);
        this.backLight.position.set(-2.5, 0.4, -1);

        this.scene = new THREE.Scene();
        this.gridScene = new THREE.Scene();

        this.axesHelper.scene = new THREE.Scene();
        this.K3DObjects = new THREE.Group();

        [this.keyLight, this.headLight, this.fillLight, this.backLight].forEach((light) => {
            self.camera.add(light);
            self.camera.add(light.target);
            // self.scene.add(new THREE.DirectionalLightHelper(light, 1.0, 0xff0000));
        });

        this.scene.add(ambientLight);
        this.scene.add(this.camera);
        this.scene.add(this.K3DObjects);

        this.cleanup = cleanup.bind(this, grids, this.gridScene);

        K3D.rebuildSceneData = rebuildSceneData.bind(this, K3D, grids, this.axesHelper);
        K3D.getSceneBoundingBox = getSceneBoundingBox.bind(this, K3D);
        K3D.refreshGrid = refreshGrid.bind(this, K3D, grids);

        K3D.rebuildSceneData().then(() => {
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

            self.backLight.visible = value > 0.0;
            self.headLight.visible = value > 0.0;
            self.fillLight.visible = value > 0.0;
            self.backLight.visible = value > 0.0;
        };

        K3D.on(K3D.events.MOUSE_MOVE, (coord) => {
            if (K3D.parameters.viewMode !== viewModes.view) {
                if (raycast.call(self, K3D, coord.x, coord.y, self.camera, false, K3D.parameters.viewMode)
                    && !K3D.autoRendering) {
                    K3D.render();
                }
            }
        });

        K3D.on(K3D.events.MOUSE_CLICK, (coord) => {
            if (K3D.parameters.viewMode !== viewModes.view) {
                if (raycast.call(self, K3D, coord.x, coord.y, self.camera, true, K3D.parameters.viewMode)
                    && !K3D.autoRendering) {
                    K3D.render();
                }
            }
        });

        K3D.on(K3D.events.RESIZED, function () {
            // update outlines
            Object.keys(grids.planes).forEach(function (axis) {
                grids.planes[axis].forEach((plane) => {
                    const objResolution = plane.obj.material.uniforms.resolution;

                    objResolution.value.x = K3D.getWorld().width;
                    objResolution.value.y = K3D.getWorld().height;
                }, this);
            }, this);
        });
    },
};
