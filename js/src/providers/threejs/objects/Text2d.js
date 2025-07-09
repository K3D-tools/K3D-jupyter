const THREE = require('three');
let katex = require('katex');
const { areAllChangesResolve } = require('../helpers/Fn');


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
    create(config, K3D) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 0;

        const text = config.text || '\\KaTeX';
        const { color } = config;
        const referencePoint = config.reference_point || 'lb';
        let { position } = config;
        const size = config.size || 1;
        const object = new THREE.Object3D();
        const { overlayDOMNode } = K3D.getWorld();
        const world = K3D.getWorld();
        let domElements = [];
        let i;

        if (position.data) {
            object.positions = position.data;
        } else {
            object.positions = position;
        }

        object.text = text;

        for (i = 0; i < object.positions.length / 2; i++) {
            const domElement = document.createElement('div');

            if (config.is_html) {
                domElement.innerHTML = Array.isArray(text) ? text[i] : text;
                domElement.style.cssText = 'pointer-events: all';
            } else {
                domElement.innerHTML = katex.renderToString(Array.isArray(text) ? text[i] : text,
                    { displayMode: true });
            }

            if (position.data) {
                position = position.data;
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

        function render() {
            domElements.forEach(function (domElement, index) {

                let x;
                let y;
                const coord = {
                    x: object.positions[index * 2] * world.width,
                    y: object.positions[index * 2 + 1] * world.height,
                };

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
                domElement.style.zIndex = 16777271;
            });
        }

        const listenersId = K3D.on(K3D.events.BEFORE_RENDER, render);
        object.domElements = domElements;

        object.onRemove = function () {
            domElements.forEach(function (domElement) {
                overlayDOMNode.removeChild(domElement);
                domElement = null;
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
                    domElement.innerHTML = katex.renderToString(text, { displayMode: true });
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
            return Promise.resolve({ json: config, obj });
        }
        return false;
    }

};

function colorToHex(color) {
    color = parseInt(color, 10) + 0x1000000;

    return `#${color.toString(16).substr(1)}`;
}
