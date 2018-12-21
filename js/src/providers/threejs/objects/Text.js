'use strict';

var katex = require('katex');

/**
 * Loader strategy to handle LaTex object
 * @method DOM
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @param {K3D}
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config, K3D) {
        config.visible = typeof(config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof(config.color) !== 'undefined' ? config.color : 0;
        config.text = typeof(config.text) !== 'undefined' ? config.text :'\\KaTeX';

        var text = config.text,
            color = config.color,
            referencePoint = config.reference_point || 'lb',
            size = config.size || 1,
            position = config.position,
            object = new THREE.Object3D(),
            domElement = document.createElement('div'),
            overlayDOMNode = K3D.getWorld().overlayDOMNode,
            listenersId;

        domElement.innerHTML = katex.renderToString(text, {displayMode: true});
        domElement.style.position = 'absolute';
        domElement.style.color = colorToHex(color);
        domElement.style.fontSize = size + 'em';

        overlayDOMNode.appendChild(domElement);

        object.position.set(position[0], position[1], position[2]);
        object.updateMatrixWorld();

        function render() {
            var coord, x, y;

            if (domElement.style.display === 'hidden') {
                return;
            }

            coord = toScreenPosition(object, K3D.getWorld());

            switch (referencePoint[0]) {
                case 'l':
                    x = coord.x + 'px';
                    break;
                case 'c':
                    x = 'calc(' + coord.x + 'px - 50%)';
                    break;
                case 'r':
                    x = 'calc(' + coord.x + 'px - 100%)';
                    break;
            }

            switch (referencePoint[1]) {
                case 't':
                    y = coord.y + 'px';
                    break;
                case 'c':
                    y = 'calc(' + coord.y + 'px - 50%)';
                    break;
                case 'b':
                    y = 'calc(' + coord.y + 'px - 100%)';
                    break;
            }

            domElement.style.transform = 'translate(' + x + ',' + y + ')';
            domElement.style.zIndex = 16777271 - Math.round(coord.z * 1e6);
        }

        listenersId = K3D.on(K3D.events.RENDERED, render);

        object.onRemove = function () {
            overlayDOMNode.removeChild(domElement);
            K3D.off(K3D.events.RENDERED, listenersId);
        };

        object.hide = function () {
            domElement.style.display = 'none';
        };

        object.show = function () {
            domElement.style.display = 'inline-block';
        };

        object.show();

        return Promise.resolve(object);
    }
};

function toScreenPosition(obj, world) {
    var vector = new THREE.Vector3(),
        widthHalf = 0.5 * world.width,
        heightHalf = 0.5 * world.height;

    obj.updateMatrixWorld();
    vector.setFromMatrixPosition(obj.matrixWorld);
    vector.project(world.camera);

    vector.x = (vector.x + 1) * widthHalf;
    vector.y = (-vector.y + 1) * heightHalf;

    return {
        x: Math.round(vector.x),
        y: Math.round(vector.y),
        z: vector.z
    };

}

function colorToHex(color) {
    color = parseInt(color, 10) + 0x1000000;

    return '#' + color.toString(16).substr(1);
}
