const THREE = require('three');
const katex = require('katex');
const { areAllChangesResolve } = require('../helpers/Fn');

/**
 * Loader strategy to handle Label object
 * @method DOM
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @param {K3D}
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config, K3D) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 0;

        const geometry = new THREE.BufferGeometry();
        const material = new THREE.MeshBasicMaterial({
            color: config.color,
        });
        const text = config.text || '\\KaTeX';
        const { color } = config;
        const maxLength = config.max_length || 1.0;
        const { position } = config;
        const size = config.size || 1;
        const object = new THREE.Group();
        const p = new THREE.Object3D();
        const line = new THREE.Line(geometry, material);
        const domElement = document.createElement('div');
        const { overlayDOMNode } = K3D.getWorld();
        const world = K3D.getWorld();

        if (config.is_html) {
            domElement.innerHTML = text;
            domElement.style.cssText = 'pointer-events: all';
        } else {
            domElement.innerHTML = katex.renderToString(text, { displayMode: true });
        }

        domElement.style.position = 'absolute';
        domElement.style.color = colorToHex(color);
        domElement.style.fontSize = `${size}em`;

        if (config.label_box) {
            domElement.style.padding = '5px';
            domElement.style.background = K3D.getWorld().targetDOMNode.style.backgroundColor;
            domElement.style.border = `1px solid ${colorToHex(color)}`;
            domElement.style.borderRadius = '10px';
        }

        overlayDOMNode.appendChild(domElement);

        p.position.set(position[0], position[1], position[2]);
        p.updateMatrixWorld();

        line.frustumCulled = false;

        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 1, 1]), 3));
        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        function render() {
            let x;
            let y;
            let v;
            let referencePoint;
            const widthHalf = 0.5 * world.width;
            const heightHalf = 0.5 * world.height;

            const coord = toScreenPosition(p, world);

            if (!coord.out) {
                domElement.style.display = 'block';

                if (config.mode === 'dynamic') {
                    let fi = Math.atan2(coord.y - heightHalf, coord.x - widthHalf);
                    let dist;
                    let fiIsOK;
                    const minDistance = 150;
                    const maxIteration = 360;
                    let
                        i = 0;

                    do {
                        dist = Math.sqrt((coord.x - widthHalf) * (coord.x - widthHalf)
                            + (coord.y - heightHalf) * (coord.y - heightHalf));

                        x = coord.x + Math.cos(fi) * Math.min(widthHalf * 0.98 - dist,
                            Math.min(widthHalf, heightHalf) * maxLength);
                        y = coord.y + Math.sin(fi) * Math.min(heightHalf * 0.98 - dist,
                            Math.min(widthHalf, heightHalf) * maxLength);

                        fiIsOK = K3D.labels.every((point) => Math.sqrt(
                            (x - point.coord.x) * (x - point.coord.x)
                            + (y - point.coord.y) * (y - point.coord.y),
                        ) > minDistance);

                        if (!fiIsOK) {
                            fi += (Math.PI / 180.0) * 0.25;
                        }

                        i++;
                    }
                    while (!fiIsOK && i < maxIteration);

                    coord.x = x;
                    coord.y = y;

                    if (fi >= -Math.PI / 4.0 && fi < Math.PI / 4.0) {
                        referencePoint = 'lc';
                        coord.x -= domElement.offsetWidth;
                    }

                    if (fi >= Math.PI / 4.0 && fi < Math.PI * 0.75) {
                        referencePoint = 'ct';
                        coord.x -= domElement.offsetWidth / 2.0;
                        coord.y -= domElement.offsetHeight;
                    }

                    if (fi >= Math.PI * 0.75 || fi < -Math.PI * 0.75) {
                        referencePoint = 'rc';
                        coord.x += domElement.offsetWidth;
                    }

                    if (fi >= -Math.PI * 0.75 && fi < -Math.PI / 4.0) {
                        referencePoint = 'cb';
                        coord.x -= domElement.offsetWidth / 2.0;
                        coord.y += domElement.offsetHeight;
                    }
                }

                if (config.mode === 'local') {
                    referencePoint = 'cb';

                    coord.y -= 0.25 * maxLength * world.height;
                }

                if (config.mode === 'side') {
                    referencePoint = 'rc';

                    y = K3D.labels.reduce((prev, val) => {
                        if (val.mode === 'side') {
                            return prev + val.domElement.offsetHeight + 10;
                        }
                        return prev;
                    }, 0);

                    coord.x = domElement.offsetWidth + 10;
                    coord.y = 10 + y + domElement.offsetHeight / 2.0;
                }

                switch (referencePoint[0]) {
                    case 'l':
                        x = `${Math.round(coord.x)}px`;
                        break;
                    case 'c':
                        x = `calc(${Math.round(coord.x)}px - 50%)`;
                        break;
                    case 'r':
                        x = `calc(${Math.round(coord.x)}px - 100%)`;
                        break;
                    default:
                        break;
                }

                switch (referencePoint[1]) {
                    case 't':
                        y = `${Math.round(coord.y)}px`;
                        break;
                    case 'c':
                        y = `calc(${Math.round(coord.y)}px - 50%)`;
                        break;
                    case 'b':
                        y = `calc(${Math.round(coord.y)}px - 100%)`;
                        break;
                    default:
                        break;
                }

                v = new THREE.Vector3(
                    (coord.x / world.width - 0.5) * 2.0,
                    -(coord.y / world.height - 0.5) * 2.0,
                    coord.z,
                );
                v.unproject(world.camera);

                geometry.attributes.position.array.set([p.position.x, p.position.y, p.position.z, v.x, v.y, v.z]);
                geometry.attributes.position.needsUpdate = true;
                geometry.computeBoundingBox();

                line.visible = true;

                domElement.style.transform = `translate(${x},${y})`;
                domElement.style.zIndex = config.on_top ? '1500' : '15';

                p.coord = coord;

                K3D.labels.push(p);
            } else {
                line.visible = false;
                domElement.style.display = 'none';
            }
        }

        const listenersId = K3D.on(K3D.events.BEFORE_RENDER, render);
        p.domElement = domElement;
        p.mode = config.mode;

        object.onRemove = function () {
            overlayDOMNode.removeChild(domElement);
            p.domElement = null;
            K3D.off(K3D.events.BEFORE_RENDER, listenersId);
        };

        object.add(p);
        object.add(line);

        render();

        return Promise.resolve(object);
    },

    update(config, changes, obj) {
        const resolvedChanges = {};

        if (typeof (changes.text) !== 'undefined' && !changes.text.timeSeries) {
            if (config.is_html) {
                obj.children[0].domElement.innerHTML = changes.text;
            } else {
                obj.children[0].domElement.innerHTML = katex.renderToString(changes.text, { displayMode: true });
            }

            resolvedChanges.text = null;
        }

        if (typeof (changes.position) !== 'undefined' && !changes.position.timeSeries) {
            obj.children[0].position.set(changes.position[0], changes.position[1], changes.position[2]);
            obj.updateMatrixWorld();

            resolvedChanges.position = null;
        }

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};

function toScreenPosition(obj, world) {
    const vector = new THREE.Vector3();

    obj.updateMatrixWorld();
    vector.setFromMatrixPosition(obj.matrixWorld);

    if (world.camera.frustum && !world.camera.frustum.containsPoint(vector)) {
        return {
            x: -10000,
            y: -10000,
            z: -10000,
            out: true,
        };
    }

    vector.project(world.camera);

    vector.x = (vector.x + 1) * 0.5 * world.width;
    vector.y = (-vector.y + 1) * 0.5 * world.height;

    return {
        x: vector.x,
        y: vector.y,
        z: vector.z,
    };
}

function colorToHex(color) {
    color = parseInt(color, 10) + 0x1000000;

    return `#${color.toString(16).substr(1)}`;
}
