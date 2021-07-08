const THREE = require('three');
const katex = require('katex');
const { areAllChangesResolve } = require('../helpers/Fn');

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
        const { position } = config;
        const size = config.size || 1;
        const object = new THREE.Object3D();
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

        object.position.set(position[0], position[1], position[2]);
        object.updateMatrixWorld();

        function render() {
            let x;
            let y;
            const coord = {
                x: object.position.x * world.width,
                y: object.position.y * world.height,
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
        }

        const listenersId = K3D.on(K3D.events.RENDERED, render);
        object.domElement = domElement;

        object.onRemove = function () {
            overlayDOMNode.removeChild(domElement);
            object.domElement = null;
            K3D.off(K3D.events.RENDERED, listenersId);
        };

        return Promise.resolve(object);
    },

    update(config, changes, obj) {
        const resolvedChanges = {};

        if (typeof (changes.text) !== 'undefined' && !changes.text.timeSeries) {
            if (config.is_html) {
                obj.domElement.innerHTML = changes.text;
                obj.domElement.style.pointerEvents = 'all';
            } else {
                obj.domElement.innerHTML = katex.renderToString(changes.text, { displayMode: true });
                obj.domElement.style.pointerEvents = 'none';
            }

            resolvedChanges.text = null;
        }

        if (typeof (changes.position) !== 'undefined' && !changes.position.timeSeries) {
            obj.position.set(changes.position[0], changes.position[1], changes.position[2]);
            obj.updateMatrixWorld();

            resolvedChanges.position = null;
        }

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },

};

function colorToHex(color) {
    color = parseInt(color, 10) + 0x1000000;

    return `#${color.toString(16).substr(1)}`;
}
