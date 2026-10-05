const THREE = require('three');
const { viewModes } = require('../../../core/lib/viewMode');
const { cameraModes } = require('../../../core/lib/cameraMode');
const { cleanup, rebuildSceneData, refreshGrid } = require('./grid');
const installLighting = require('./lighting');

function getSceneBoundingBox(K3D) {
    const sceneBoundingBox = new THREE.Box3();
    let objectBoundingBox;
    const world = K3D.getWorld();

    Object.keys(world.ObjectsListJson).forEach((K3DIdentifier) => {
        const k3dObject = world.ObjectsById[K3DIdentifier];

        if (!k3dObject) {
            return;
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

                // A box with NaN/Infinity (e.g. from NaN-separated line vertices) would poison
                // the union and end up as NaN camera near/far planes. An empty box is fine —
                // union() ignores it.
                if (!objectBoundingBox.isEmpty()
                    && !(Number.isFinite(objectBoundingBox.min.x) && Number.isFinite(objectBoundingBox.min.y)
                        && Number.isFinite(objectBoundingBox.min.z) && Number.isFinite(objectBoundingBox.max.x)
                        && Number.isFinite(objectBoundingBox.max.y) && Number.isFinite(objectBoundingBox.max.z))) {
                    return;
                }

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

function raycast(K3D, x, y, camera, click, viewMode) {
    const meshes = [];
    let intersects = [];
    let needRender = false;

    this.raycaster.setFromCamera(new THREE.Vector2(x, y), camera);

    // traverseVisible, not traverse: an object hidden by visible - including a time series of
    // it - is not on screen, so it must not answer a click nor steal one from what is behind it
    this.K3DObjects.traverseVisible((object) => {
        if (object.interactions) {
            if (object.geometry && object.geometry.attributes.position.count === 0) {
                return;
            }

            if (object.interactions.intersect) {
                intersects = intersects.concat(object.interactions.intersect(this.raycaster));
            } else {
                meshes.push(object);
            }
        }
    });

    if (meshes.length > 0) {
        intersects = intersects.concat(this.raycaster.intersectObjects(meshes));
    }

    if (intersects.length > 1) {
        intersects.sort((a, b) => a.distance - b.distance);
    }

    if (intersects.length > 0) {
        const intersect = intersects[0];
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
        const grids = {
            planes: {},
            labelsOnPlanes: {},
        };
        const self = this;
        this.lastMouseCoord = null;

        this.lights = [];
        this.raycaster = new THREE.Raycaster();
        this.raycaster.firstHitOnly = true;

        this.scene = new THREE.Scene();
        this.gridScene = new THREE.Scene();

        this.axesHelper.scene = new THREE.Scene();
        this.K3DObjects = new THREE.Group();

        installLighting(K3D, self);

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

        function cb(click, coord) {
            if (typeof (coord) === 'undefined') {
                if (self.lastMouseCoord === null) {
                    return;
                }
                coord = self.lastMouseCoord;
            }
            self.lastMouseCoord = coord;

            if (K3D.parameters.viewMode !== viewModes.view) {
                if (K3D.parameters.cameraMode === cameraModes.volumeSides) {
                    if (coord.x < 0 && coord.y > 0) {
                        K3D.getWorld().controls.beforeRender(0);
                        if (raycast.call(
                            self,
                            K3D,
                            (coord.x + 0.5) * 2,
                            (coord.y - 0.5) * 2,
                            self.camera,
                            click,
                            K3D.parameters.viewMode,
                        )) {
                            K3D.render();
                        }
                        K3D.getWorld().controls.afterRender(0);
                    } else if (coord.x < 0 && coord.y < 0) {
                        K3D.getWorld().controls.beforeRender(1);
                        if (raycast.call(
                            self,
                            K3D,
                            (coord.x + 0.5) * 2,
                            (1.0 + coord.y * 2.0),
                            self.camera,
                            click,
                            K3D.parameters.viewMode,
                        )) {
                            K3D.render();
                        }
                        K3D.getWorld().controls.afterRender(1);
                    } else if (coord.x > 0 && coord.y > 0) {
                        K3D.getWorld().controls.beforeRender(2);
                        if (raycast.call(
                            self,
                            K3D,
                            (coord.x * 2.0 - 1.0),
                            (coord.y - 0.5) * 2,
                            self.camera,
                            click,
                            K3D.parameters.viewMode,
                        )) {
                            K3D.render();
                        }
                        K3D.getWorld().controls.afterRender(2);
                    } else if (coord.x > 0 && coord.y < 0) {
                        K3D.getWorld().controls.beforeRender(3);
                        if (raycast.call(
                            self,
                            K3D,
                            (coord.x * 2.0 - 1.0),
                            (1.0 + coord.y * 2.0),
                            self.camera,
                            click,
                            K3D.parameters.viewMode,
                        )) {
                            K3D.render();
                        }
                        K3D.getWorld().controls.afterRender(3);
                    }
                } else if (raycast.call(self, K3D, coord.x, coord.y, self.camera, click, K3D.parameters.viewMode)) {
                    K3D.render();
                }
            }
        }

        K3D.on(K3D.events.MOUSE_LEAVE, () => {
            // without this the RENDERED pass below keeps hovering the last position under a
            // cursor that is no longer over the canvas
            self.lastMouseCoord = null;
        });

        K3D.on(K3D.events.MOUSE_MOVE, cb.bind(this, false));
        K3D.on(K3D.events.MOUSE_CLICK, cb.bind(this, true));
        K3D.on(K3D.events.RENDERED, function () {
            setTimeout(cb.bind(this, false), 0);
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
