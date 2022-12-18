const _ = require('../../lodash');

function detachWindowGUI(gui, K3D) {
    let newWindow;
    let intervalID;
    const originalDom = K3D.getWorld().targetDOMNode;
    const detachedDom = document.createElement('div');
    const detachedDomText = 'K3D works in detached mode. Please close window or click on this text to attach it here.';

    function removeByTagName(DOM, tag) {
        const elements = DOM.getElementsByTagName(tag);

        while (elements[0]) {
            elements[0].parentNode.removeChild(elements[0]);
        }
    }

    function reinitializeK3D(DOM) {
        const world = K3D.getWorld();
        const newK3D = new K3D.constructor(K3D.Provider, DOM, K3D.parameters);
        const objects = world.K3DObjects.children.reduce((prev, object) => {
            prev.push(world.ObjectsListJson[object.K3DIdentifier]);

            return prev;
        }, []);

        K3D.disable();
        newK3D.load({objects});
        newK3D.setCamera(K3D.getWorld().controls.getCameraArray());

        _.assign(K3D, newK3D);
    }

    function attach() {
        if (newWindow) {
            newWindow.close();
            newWindow = null;
            originalDom.removeChild(detachedDom);
            reinitializeK3D(originalDom);
        }
    }

    function checkWindow() {
        if (newWindow && newWindow.closed) {
            clearInterval(intervalID);
            attach();
        }
    }

    detachedDom.className = 'detachedInfo';
    detachedDom.innerHTML = detachedDomText;

    detachedDom.style.cssText = [
        'cursor: pointer',
        'padding: 2em',
    ].join(';');

    const obj = {
        detachWidget() {
            newWindow = window.open('', '_blank', 'width=800,height=600,resizable=1');
            newWindow.document.body.innerHTML = require('./helpers/detachedWindowHtml');

            // copy css
            const styles = document.getElementsByTagName('style');

            for (let i = 0; i < styles.length; i++) {
                newWindow.document.body.appendChild(
                    styles[i].cloneNode(true),
                );
            }

            setTimeout(() => {
                reinitializeK3D(newWindow.document.getElementById('canvasTarget'));

                removeByTagName(originalDom, 'canvas');
                removeByTagName(originalDom, 'div');
                removeByTagName(originalDom, 'svg');

                originalDom.appendChild(detachedDom);
            }, 100);

            newWindow.opener.addEventListener('unload', () => {
                if (newWindow) {
                    newWindow.close();
                }
            }, false);

            intervalID = setInterval(checkWindow, 500);
        },
    };

    gui.add(obj, 'detachWidget').name('Detach widget');

    detachedDom.addEventListener('click', () => {
        attach();
    }, false);
}

module.exports = detachWindowGUI;
