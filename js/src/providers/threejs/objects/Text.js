const THREE = require('three');
const katex = require('katex');

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

        const { text } = config;
        const { color } = config;
        const referencePoint = config.reference_point || 'lb';
        const size = config.size || 1;
        const { position } = config;
        const object = new THREE.Object3D();
        const domElement = document.createElement('div');
        const world = K3D.getWorld();
        const { overlayDOMNode } = world;

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
        object.text = text;
        object.updateMatrixWorld();

        function render() {
            let coord;
            let x;
            let y;

            if (domElement.style.display === 'hidden') {
                return;
            }

            if (axesHelper) {
                coord = toScreenPosition(object, {
                    width: axesHelper.width,
                    height: axesHelper.height,
                    offsetX: world.width - axesHelper.width,
                    offsetY: world.height - axesHelper.height,
                },
                axesHelper.camera);
            } else {
                coord = toScreenPosition(object, {
                    width: world.width,
                    height: world.height,
                    offsetX: 0,
                    offsetY: 0,
                },
                world.camera);
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
        }

        const listenersId = K3D.on(K3D.events.RENDERED, render);

        object.onRemove = function () {
            if (overlayDOMNode.contains(domElement)) {
                overlayDOMNode.removeChild(domElement);
            }

            K3D.off(K3D.events.RENDERED, listenersId);
        };

        object.hide = function () {
            domElement.style.display = 'none';
        };

        object.show = function () {
            domElement.style.display = 'block';
        };

        object.show();

        return Promise.resolve(object);
    },
};

function toScreenPosition(obj, viewport, camera) {
    const vector = new THREE.Vector3();
    const widthHalf = 0.5 * viewport.width;
    const heightHalf = 0.5 * viewport.height;

    obj.updateMatrixWorld();
    vector.setFromMatrixPosition(obj.matrixWorld);

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
    color = parseInt(color, 10) + 0x1000000;

    return `#${color.toString(16).substr(1)}`;
}
