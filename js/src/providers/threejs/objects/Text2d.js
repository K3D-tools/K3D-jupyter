'use strict';

var THREE = require('three'),
    katex = require('katex'),
    areAllChangesResolve = require('./../helpers/Fn').areAllChangesResolve;

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
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 0;

        var text = config.text || '\\KaTeX',
            color = config.color,
            referencePoint = config.reference_point || 'lb',
            position = config.position,
            size = config.size || 1,
            object = new THREE.Object3D(),
            domElement = document.createElement('div'),
            overlayDOMNode = K3D.getWorld().overlayDOMNode,
            listenersId,
            world = K3D.getWorld();

        domElement.innerHTML = katex.renderToString(text, {displayMode: true});
        domElement.style.position = 'absolute';
        domElement.style.color = colorToHex(color);
        domElement.style.fontSize = size + 'em';

        overlayDOMNode.appendChild(domElement);

        object.position.set(position[0], position[1], position[2]);
        object.updateMatrixWorld();

        function render() {
            var x, y, coord = {
                x: position[0] * world.width,
                y: position[1] * world.height
            };

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
            domElement.style.zIndex = 16777271;
        }

        listenersId = K3D.on(K3D.events.RENDERED, render);
        object.domElement = domElement;

        object.onRemove = function () {
            overlayDOMNode.removeChild(domElement);
            object.domElement = null;
            K3D.off(K3D.events.RENDERED, listenersId);
        };

        return Promise.resolve(object);
    },

    update: function (config, changes, obj) {
        if (typeof(changes.text) !== 'undefined' && !changes.text.timeSeries) {
            obj.domElement.innerHTML = katex.renderToString(changes.text, {displayMode: true});

            changes.text = null;
        }

        if (typeof(changes.position) !== 'undefined' && !changes.position.timeSeries) {
            obj.position.set(changes.position[0], changes.position[1], changes.position[2]);
            obj.updateMatrixWorld();

            changes.position = null;
        }

        if (areAllChangesResolve(changes)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }

};

function colorToHex(color) {
    color = parseInt(color, 10) + 0x1000000;

    return '#' + color.toString(16).substr(1);
}
