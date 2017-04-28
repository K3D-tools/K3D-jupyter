'use strict';

function createGrids(scene) {
    var decGrid, unitGrid;

    decGrid = new THREE.GridHelper(100, 100);
    decGrid.rotation.x = THREE.Math.degToRad(90);
    decGrid.material.opacity = 0.25;
    decGrid.material.transparent = true;

    unitGrid = new THREE.GridHelper(100, 10);
    unitGrid.rotation.x = THREE.Math.degToRad(90);
    unitGrid.material.opacity = 0.75;
    unitGrid.material.transparent = true;

    scene.add(decGrid);
    scene.add(unitGrid);
}

function rebuildOctree() {
    /*jshint validthis:true */
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
module.exports = function (K3D) {
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

    createGrids(this.scene);

    K3D.raycast = raycast.bind(this);
    K3D.rebuildOctree = rebuildOctree.bind(this);

    K3D.rebuildOctree();
};
