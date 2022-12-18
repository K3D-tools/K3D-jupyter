const THREE = require('three');
let katex = require('katex');
const {areAllChangesResolve} = require("../helpers/Fn");

katex = katex.default || katex;

/**
 * Loader strategy to handle LaTex object
 * @method DOM
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @param {K3D}
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config, K3D, axesHelper) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 0;
        config.text = typeof (config.text) !== 'undefined' ? config.text : '\\KaTeX';

        const modelMatrix = new THREE.Matrix4();
        const {text} = config;
        const {color} = config;
        const referencePoint = config.reference_point || 'lb';
        const size = config.size || 1;
        let {position} = config;
        const object = new THREE.Object3D();
        const world = K3D.getWorld();
        const {overlayDOMNode} = world;
        let domElements = [];
        let i;

        if (position.data) {
            object.positions = position.data;
        } else {
            object.positions = position;
        }

        object.text = text;

        for (i = 0; i < object.positions.length / 3; i++) {
            const domElement = document.createElement('div');

            if (config.is_html) {
                domElement.innerHTML = Array.isArray(text) ? text[i] : text;
                domElement.style.cssText = 'pointer-events: all';
            } else {
                domElement.innerHTML = katex.renderToString(Array.isArray(text) ? text[i] : text, {displayMode: true});
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
            domElements.push(domElement);
        }

        if (config.model_matrix) {
            modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
            object.applyMatrix4(modelMatrix);
        }

        object.updateMatrixWorld();

        function render() {
            domElements.forEach(function (domElement, index) {
                let coord;
                let x;
                let y;

                if (domElement.style.display === 'hidden') {
                    return;
                }

                if (axesHelper) {
                    coord = toScreenPosition(
                        object,
                        index,
                        {
                            width: axesHelper.width,
                            height: axesHelper.height,
                            offsetX: world.width - axesHelper.width,
                            offsetY: world.height - axesHelper.height,
                        },
                        axesHelper.camera,
                    );
                } else {
                    coord = toScreenPosition(
                        object,
                        index,
                        {
                            width: world.width,
                            height: world.height,
                            offsetX: 0,
                            offsetY: 0,
                        },
                        world.camera,
                    );
                }

                switch (referencePoint[0]) {
                    case 'l':
                        x = `${coord.x}px`;
                        break;
                    case 'c':
                        x = `calc(${coord.x}px - 50%)`;
                        break;
                    case 'r':
                        x = `calc(${coord.x}px - 100%)`;
                        break;
                    default:
                        break;
                }

                switch (referencePoint[1]) {
                    case 't':
                        y = `${coord.y}px`;
                        break;
                    case 'c':
                        y = `calc(${coord.y}px - 50%)`;
                        break;
                    case 'b':
                        y = `calc(${coord.y}px - 100%)`;
                        break;
                    default:
                        break;
                }

                domElement.style.transform = `translate(${x},${y})`;
                domElement.style.zIndex = config.on_top ? 16777271 - Math.round(coord.z * 1e6) : '15';
            });
        }

        const listenersId = K3D.on(K3D.events.BEFORE_RENDER, render);
        object.domElements = domElements;

        object.onRemove = function () {
            domElements.forEach(function (domElement) {
                if (overlayDOMNode.contains(domElement)) {
                    overlayDOMNode.removeChild(domElement);
                    domElement = null;
                }
            });

            domElements = [];

            K3D.off(K3D.events.BEFORE_RENDER, listenersId);
        };

        object.hide = function () {
            domElements.forEach(function (domElement) {
                domElement.style.display = 'none';
            });
        };

        object.show = function () {
            domElements.forEach(function (domElement) {
                domElement.style.display = 'block';
            });
        };

        object.boundingBox = new THREE.Box3().setFromArray(object.positions);

        object.show();

        return Promise.resolve(object);
    },
    update(config, changes, obj) {
        const resolvedChanges = {};

        if (typeof (changes.text) !== 'undefined' && !changes.text.timeSeries) {

            obj.domElements.forEach(function (domElement, i) {
                let text = Array.isArray(changes.text) ? changes.text[i] : changes.text;

                if (config.is_html) {
                    domElement.innerHTML = text;
                    domElement.style.pointerEvents = 'all';
                } else {
                    domElement.innerHTML = katex.renderToString(text, {displayMode: true});
                    domElement.style.pointerEvents = 'none';
                }
            });

            resolvedChanges.text = null;
        }

        if (typeof (changes.position) !== 'undefined' && !changes.position.timeSeries) {
            obj.positions = changes.position.data || changes.position;

            resolvedChanges.position = null;
        }

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj});
        }
        return false;
    }
};

function toScreenPosition(obj, index, viewport, camera) {
    const vector = new THREE.Vector3();
    const widthHalf = 0.5 * viewport.width;
    const heightHalf = 0.5 * viewport.height;

    vector.fromArray(obj.positions, index * 3);
    vector.applyMatrix4(obj.matrixWorld);

    if (camera.frustum && !camera.frustum.containsPoint(vector)) {
        return {
            x: -1000,
            y: -1000,
            z: -1000,
        };
    }

    vector.project(camera);

    vector.x = (vector.x + 1) * widthHalf + viewport.offsetX;
    vector.y = (-vector.y + 1) * heightHalf + viewport.offsetY;

    return {
        x: Math.round(vector.x),
        y: Math.round(vector.y),
        z: vector.z,
    };
}

function colorToHex(color) {
    return `#${new THREE.Color(color).getHexString()}`;
}
