'use strict';

var THREE = require('three'),
    katex = require('katex'),
    areAllChangesResolve = require('./../helpers/Fn').areAllChangesResolve;

/**
 * Loader strategy to handle Label object
 * @method DOM
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @param {K3D}
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config, K3D) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 0;

        var geometry = new THREE.BufferGeometry(),
            material = new THREE.MeshBasicMaterial({
                color: config.color
            }),
            text = config.text || '\\KaTeX',
            color = config.color,
            maxLength = config.max_length || 1.0,
            position = config.position,
            size = config.size || 1,
            object = new THREE.Group(),
            p = new THREE.Object3D(),
            line = new THREE.Line(geometry, material),
            domElement = document.createElement('div'),
            overlayDOMNode = K3D.getWorld().overlayDOMNode,
            listenersId,
            world = K3D.getWorld();

        if (config.is_html) {
            domElement.innerHTML = text;
            domElement.style.cssText = 'pointer-events: all';
        } else {
            domElement.innerHTML = katex.renderToString(text, {displayMode: true});
        }

        domElement.style.position = 'absolute';
        domElement.style.color = colorToHex(color);
        domElement.style.fontSize = size + 'em';

        if (config.label_box) {
            domElement.style.padding = '5px';
            domElement.style.background = K3D.getWorld().targetDOMNode.style.backgroundColor;
            domElement.style.border = '1px solid ' + colorToHex(color);
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
            var coord, x, y, v, referencePoint,
                widthHalf = 0.5 * world.width,
                heightHalf = 0.5 * world.height;

            coord = toScreenPosition(p, world);

            if (!coord.out) {
                domElement.style.display = 'block';

                if (config.mode === 'dynamic') {
                    var fi = Math.atan2(coord.y - heightHalf, coord.x - widthHalf), dist, fiIsOK,
                        minDistance = 150, maxIteration = 360, i = 0;

                    do {
                        dist = Math.sqrt((coord.x - widthHalf) * (coord.x - widthHalf) +
                                         (coord.y - heightHalf) * (coord.y - heightHalf));

                        x = coord.x + Math.cos(fi) * Math.min(widthHalf * 0.98 - dist,
                            Math.min(widthHalf, heightHalf) * maxLength);
                        y = coord.y + Math.sin(fi) * Math.min(heightHalf * 0.98 - dist,
                            Math.min(widthHalf, heightHalf) * maxLength);

                        fiIsOK = K3D.labels.every(function (p) {
                            return Math.sqrt(
                                (x - p.coord.x) * (x - p.coord.x) +
                                (y - p.coord.y) * (y - p.coord.y)) > minDistance;
                        });

                        if (!fiIsOK) {
                            fi += Math.PI / 180.0 * 0.25;
                        }

                        i++;
                    } while (!fiIsOK && i < maxIteration);

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

                    y = K3D.labels.reduce(function (prev, val) {
                        if (val.mode === 'side') {
                            return prev + val.domElement.offsetHeight + 10;
                        } else {
                            return prev;
                        }
                    }, 0);

                    coord.x = domElement.offsetWidth + 10;
                    coord.y = 10 + y + domElement.offsetHeight / 2.0;
                }

                switch (referencePoint[0]) {
                    case 'l':
                        x = Math.round(coord.x) + 'px';
                        break;
                    case 'c':
                        x = 'calc(' + Math.round(coord.x) + 'px - 50%)';
                        break;
                    case 'r':
                        x = 'calc(' + Math.round(coord.x) + 'px - 100%)';
                        break;
                }

                switch (referencePoint[1]) {
                    case 't':
                        y = Math.round(coord.y) + 'px';
                        break;
                    case 'c':
                        y = 'calc(' + Math.round(coord.y) + 'px - 50%)';
                        break;
                    case 'b':
                        y = 'calc(' + Math.round(coord.y) + 'px - 100%)';
                        break;
                }

                v = new THREE.Vector3(
                    (coord.x / world.width - 0.5) * 2.0,
                    -(coord.y / world.height - 0.5) * 2.0,
                    coord.z);
                v.unproject(world.camera);

                geometry.attributes.position.array.set([p.position.x, p.position.y, p.position.z, v.x, v.y, v.z]);
                geometry.attributes.position.needsUpdate = true;
                geometry.computeBoundingBox();

                line.visible = true;

                domElement.style.transform = 'translate(' + x + ',' + y + ')';
                domElement.style.zIndex = config.on_top ? '1500' : '5';

                p.coord = coord;

                K3D.labels.push(p);
            } else {
                line.visible = false;
                domElement.style.display = 'none';
            }
        }

        listenersId = K3D.on(K3D.events.BEFORE_RENDER, render);
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

    update: function (config, changes, obj) {
        var resolvedChanges = {};

        if (typeof(changes.text) !== 'undefined' && !changes.text.timeSeries) {
            if (config.is_html) {
                obj.children[0].domElement.innerHTML = changes.text;
            } else {
                obj.children[0].domElement.innerHTML = katex.renderToString(changes.text, {displayMode: true});
            }

            resolvedChanges.text = null;
        }

        if (typeof(changes.position) !== 'undefined' && !changes.position.timeSeries) {
            obj.children[0].position.set(changes.position[0], changes.position[1], changes.position[2]);
            obj.updateMatrixWorld();

            resolvedChanges.position = null;
        }

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
};

function toScreenPosition(obj, world) {
    var vector = new THREE.Vector3();

    obj.updateMatrixWorld();
    vector.setFromMatrixPosition(obj.matrixWorld);

    if (world.camera.frustum && !world.camera.frustum.containsPoint(vector)) {
        return {
            x: -10000,
            y: -10000,
            z: -10000,
            out: true
        };
    }

    vector.project(world.camera);

    vector.x = (vector.x + 1) * 0.5 * world.width;
    vector.y = (-vector.y + 1) * 0.5 * world.height;

    return {
        x: vector.x,
        y: vector.y,
        z: vector.z
    };
}

function colorToHex(color) {
    color = parseInt(color, 10) + 0x1000000;

    return '#' + color.toString(16).substr(1);
}
