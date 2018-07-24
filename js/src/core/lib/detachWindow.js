'use strict';

var _ = require('lodash');

function detachWindowGUI(gui, K3D) {
    var newWindow,
        intervalID,
        originalDom = K3D.getWorld().targetDOMNode,
        detachedDom = document.createElement('div'),
        detachedDomText = 'K3D works in detached mode. Please close window or click on this text to attach it here.',
        obj;

    function removeByTagName(DOM, tag) {
        var elements = DOM.getElementsByTagName(tag);

        while (elements[0]) {
            elements[0].parentNode.removeChild(elements[0]);
        }
    }

    function reinitializeK3D(DOM) {
        var newK3D, objects, world = K3D.getWorld();
        newK3D = new K3D.constructor(K3D.Provider, DOM, K3D.parameters);

        objects = world.K3DObjects.children.reduce(function (prev, object) {
            prev.push(world.ObjectsListJson[object.K3DIdentifier]);

            return prev;
        }, []);

        K3D.disable();
        newK3D.load({objects: objects});
        newK3D.setCamera(K3D.getWorld().controls.getCameraArray());

        _.assign(K3D, newK3D);
    }

    function checkWindow() {
        if (newWindow && newWindow.closed) {
            clearInterval(intervalID);
            attach();
        }
    }

    function attach() {
        if (newWindow) {
            newWindow.close();
            newWindow = null;
            originalDom.removeChild(detachedDom);
            reinitializeK3D(originalDom);
        }
    }

    detachedDom.className = 'detachedInfo';
    detachedDom.innerHTML = detachedDomText;

    detachedDom.style.cssText = [
        'cursor: pointer',
        'padding: 2em'
    ].join(';');

    obj = {
        detachWidget: function () {
            newWindow = window.open('', '_blank', 'width=800,height=600,resizable=1');
            newWindow.document.body.innerHTML = require('./helpers/detachedWindowHtml');

            // copy css
            ['k3d-katex', 'k3d-style', 'k3d-dat.gui'].forEach(function (id) {
                newWindow.document.body.appendChild(
                    window.document.getElementById(id).cloneNode(true)
                );
            });

            setTimeout(function () {
                reinitializeK3D(newWindow.document.getElementById('canvasTarget'));

                removeByTagName(originalDom, 'canvas');
                removeByTagName(originalDom, 'div');

                originalDom.appendChild(detachedDom);
            }, 100);

            newWindow.opener.addEventListener('unload', function () {
                if (newWindow) {
                    newWindow.close();
                }
            }, false);

            intervalID = setInterval(checkWindow, 500);
        }
    };

    gui.add(obj, 'detachWidget').name('Detach widget');

    detachedDom.addEventListener('click', function () {
        attach();
    }, false);
}

module.exports = detachWindowGUI;
