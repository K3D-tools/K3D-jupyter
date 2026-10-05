const THREE = require('three');
const _ = require('../../../lodash');
const Text = require('../objects/Text');
const Vectors = require('../objects/Vectors');
const MeshLine = require('../helpers/THREE.MeshLine')(THREE);
const { pow10ceil } = require('../../../core/lib/helpers/math');
const { recalculateFrustum } = require('../helpers/Fn');

// The grid, its tick labels and the axes helper, rebuilt from the scene's bounding box - which is
// also what decides the camera's clipping range.
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
            axesHelper.labelColor = K3D.parameters.labelColor;
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

            if (plane.obj) {
                if (plane.obj.geometry) {
                    plane.obj.geometry.dispose();
                }

                if (plane.obj.material) {
                    plane.obj.material.dispose();
                }
            }

            delete plane.obj;
        });
    });

    Object.keys(grids.labelsOnPlanes).forEach((key) => {
        grids.labelsOnPlanes[key].labels.forEach((label) => {
            label.onRemove();
        });
    });
}

function rebuildSceneData(K3D, grids, axesHelper, force) {
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
    let size;
    let majorScale;
    let minorScale;
    let i;
    let sceneBoundingBox = new THREE.Box3().setFromArray(K3D.parameters.grid);
    const unitVectors = {
        x: new THREE.Vector3(1.0, 0.0, 0.0),
        y: new THREE.Vector3(0.0, 1.0, 0.0),
        z: new THREE.Vector3(0.0, 0.0, 1.0),
    };
    const cornerToLabeledEdges = {
        '+x+y+z': ['+x-y', '+x-z', '-x+y', '+y-z', '-x+z', '-y+z'],
        '+x+y-z': ['+x-y', '+x+z', '-x+y', '+y+z', '-x-z', '-y-z'],
        '+x-y+z': ['+x+y', '+x-z', '-x-y', '-y-z', '-x+z', '+y+z'],
        '+x-y-z': ['+x+y', '+x+z', '-x-y', '-y+z', '-x-z', '+y-z'],
        '-x+y+z': ['-x-y', '-x-z', '+x+y', '+y-z', '+x+z', '-y+z'],
        '-x+y-z': ['-x-y', '-x+z', '+x+y', '+y+z', '+x-z', '-y-z'],
        '-x-y+z': ['-x+y', '-x-z', '+x-y', '-y-z', '+x+z', '+y+z'],
        '-x-y-z': ['-x+y', '-x+z', '+x-y', '-y+z', '+x-z', '+y-z'],
    };
    const labelsShiftMap = ['x', 'x', 'y', 'y', 'z', 'z'];

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

        // the letters carry the label colour in their style, set once when they were made
        updateAxesHelper |= K3D.parameters.labelColor !== axesHelper.labelColor;
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

        // only with auto fit: otherwise the box stays the grid the user asked for
        if (K3D.parameters.gridAutoFit) {
            sceneBoundingBox = K3D.getSceneBoundingBox() || sceneBoundingBox;
        }

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
            Math.floor(sceneBoundingBox.min.x / minorScale) * minorScale,
            Math.floor(sceneBoundingBox.min.y / minorScale) * minorScale,
            Math.floor(sceneBoundingBox.min.z / minorScale) * minorScale,
        );

        sceneBoundingBox.max = new THREE.Vector3(
            Math.ceil(sceneBoundingBox.max.x / minorScale) * minorScale,
            Math.ceil(sceneBoundingBox.max.y / minorScale) * minorScale,
            Math.ceil(sceneBoundingBox.max.z / minorScale) * minorScale,
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

        originalEdges = generateEdgesPoints(sceneBoundingBox);

        // create labels for ticks - iterate over all 8 corners of box

        for (i = 0; i < 8; i++) {
            const corner = `${i & 0x01 ? '-' : '+'}x${i & 0x02 ? '-' : '+'}y${i & 0x04 ? '-' : '+'}z`;

            grids.labelsOnPlanes[corner] = {};
            grids.labelsOnPlanes[corner].labels = [];

            cornerToLabeledEdges[corner].forEach((edge, index) => {
                let j;
                let p;
                let label;
                const iterateAxis = _.difference(['x', 'y', 'z'], edge.replace(/[^xyz]/g, '').split(''))[0];

                let deltaPosition = unitVectors[iterateAxis].clone().multiplyScalar(majorScale);
                let iterationCount = size[iterateAxis] / majorScale;
                const line = originalEdges[edge];

                if (iterationCount <= 2) {
                    const originalIterationCount = iterationCount;

                    iterationCount = Math.max(originalIterationCount * 5, 2);
                    deltaPosition = unitVectors[iterateAxis].clone()
                        .multiplyScalar((originalIterationCount * majorScale) / iterationCount);
                }

                const labelShiftDirection = corner[Math.floor(index / 2) * 2] === '+';

                // axis ticks labels
                for (j = 1; j <= iterationCount - 1; j++) {
                    p = line[0].clone().add(deltaPosition.clone().multiplyScalar(j)).add(
                        unitVectors[labelsShiftMap[index]].clone()
                            .multiplyScalar(minorScale * (labelShiftDirection ? 1 : -1)),
                    );

                    label = Text.create({
                        position: p.toArray(),
                        reference_point: 'cc',
                        color: labelColor,
                        text: parseFloat((p[iterateAxis]).toFixed(10)).toString(),
                        size: 0.75,
                    }, K3D);

                    promises.push(label.then((obj) => {
                        grids.labelsOnPlanes[corner].labels.push(obj);
                        // born hidden: refreshGrid decides which corner shows, and may have run already
                        obj.hide();
                    }));
                }

                // axis label
                p = (new THREE.Vector3()).lerpVectors(line[0], line[1], 0.5).add(
                    unitVectors[labelsShiftMap[index]].clone()
                        .multiplyScalar(minorScale * 2.0 * (labelShiftDirection ? 1 : -1)),
                );

                const axisLabel = Text.create({
                    position: p.toArray(),
                    reference_point: 'cc',
                    color: labelColor,
                    text: K3D.parameters.axes[['x', 'y', 'z'].indexOf(iterateAxis)],
                    size: 1.0,
                }, K3D);

                promises.push(axisLabel.then((obj) => {
                    grids.labelsOnPlanes[corner].labels.push(obj);
                    obj.hide();
                }));
            });
        }

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
                }, K3D);

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

    const camDistance = (fullSceneDiameter / 2.0) / Math.sin(THREE.MathUtils.degToRad(K3D.parameters.cameraFov / 2.0));

    this.camera.far = (camDistance + fullSceneDiameter / 2) * 5.0;
    this.camera.near = fullSceneDiameter * 0.0001;
    this.camera.updateProjectionMatrix();

    // the frustum clips the DOM labels, and it was computed with the previous far plane
    recalculateFrustum(this.camera);

    rebuildSceneDataPromises = promises;

    return Promise.all(promises).then((v) => {
        rebuildSceneDataPromises = null;
        // the new labels are hidden until the grid says which of them show
        refreshGrid.call(this, K3D, grids);
        return v;
    });
}

function refreshGrid(K3D, grids) {
    let currentCorner = '';
    const cameraDirection = new THREE.Vector3();

    this.camera.getWorldDirection(cameraDirection);

    // a rebuild can settle after disable() has torn the grid down
    if (K3D.disabling) {
        return;
    }

    Object.keys(grids.planes).forEach((axis) => {
        const dot1 = grids.planes[axis][0].normal.dot(cameraDirection);
        const dot2 = grids.planes[axis][1].normal.dot(cameraDirection);

        grids.planes[axis][0].obj.visible = dot1 <= dot2 && K3D.parameters.gridVisible;
        grids.planes[axis][1].obj.visible = dot1 > dot2 && K3D.parameters.gridVisible;

        currentCorner += (dot1 <= dot2 ? '-' : '+') + axis;
    }, this);

    Object.keys(grids.labelsOnPlanes).forEach((corner) => {
        grids.labelsOnPlanes[corner].labels.forEach((label) => {
            if (corner === currentCorner && K3D.parameters.gridVisible) {
                label.show();
            } else {
                label.hide();
            }
        });
    });
}

module.exports = {
    cleanup,
    rebuildSceneData,
    refreshGrid,
};
