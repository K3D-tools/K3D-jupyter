'use strict';

var katex = require('katex');

/**
 * Loader strategy to handle LaTex object
 * @method DOM
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {K3D.Config} config all configurations params from JSON
 * @param {K3D}
 * @return {Object} 3D object ready to render
 */
module.exports = function (config, K3D) {

    var text = config.get('text', '\\KaTeX'),
        color = config.get('color', 0),
        referencePoint = config.get('referencePoint', 'lb'),
    // colorStroke = config.get('color', 0xffffff),
        size = config.get('size', 1),
        position = config.get('position'),
        object = new THREE.Object3D(),
        domElement = document.createElement('div'),
        overlayDOMNode = K3D.getWorld().overlayDOMNode;

    // domElement.style.textShadow = '-1px -1px 0 ' + colorToHex(colorStroke) + ',' +
    //     '1px -1px 0 ' + colorToHex(colorStroke) + ',' +
    //     '-1px 1px 0 ' + colorToHex(colorStroke) + ',' +
    //     '1px 1px 0 ' + colorToHex(colorStroke);

    domElement.innerHTML = katex.renderToString(text, {displayMode: true});
    domElement.style.display = 'inline-block';
    domElement.style.position = 'absolute';
    domElement.style.color = colorToHex(color);
    domElement.style.fontSize = size + 'em';

    overlayDOMNode.appendChild(domElement);

    object.position.set(position[0], position[1], position[2]);
    object.updateMatrixWorld();

    function render() {
        var coord = toScreenPosition(object, K3D.getWorld()), x, y;

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

    K3D.on(K3D.events.RENDERED, render);

    render();

    object.onRemove = function () {
        overlayDOMNode.removeChild(domElement);
    };

    return Promise.resolve(object);
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
