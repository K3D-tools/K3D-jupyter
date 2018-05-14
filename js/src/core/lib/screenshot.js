'use strict';

var rasterizeHTML = require('rasterizehtml');

function getScreenshot(K3D) {

    // var screenshot = K3D.getWorld().renderer.domElement.toDataURL();
    // return screenshot;

    return new Promise(function (resolve, reject) {
        var finalCanvas = document.createElement('canvas'),
            canvas3d = K3D.getWorld().renderer.domElement,
            renderPromise;

        finalCanvas.width = canvas3d.width;
        finalCanvas.height = canvas3d.height;

        renderPromise = rasterizeHTML.drawHTML('<style>body{margin:0;}</style>' +
            document.getElementById('k3d-katex').outerHTML +
            document.getElementById('k3d-style').outerHTML +
            K3D.getWorld().overlayDOMNode.outerHTML, finalCanvas);

        renderPromise.then(function (result) {
            K3D.getWorld().render();
            finalCanvas.getContext('2d').drawImage(canvas3d, 0, 0);
            finalCanvas.getContext('2d').drawImage(result.image, 0, 0);
            resolve(finalCanvas);
        }, function (e) {
            reject(e);
        });
    });
}

function screenshotGUI(gui, K3D) {
    var obj = {
        screenshot: function () {
            getScreenshot(K3D).then(function (canvas) {
                var element = document.createElement('a');

                element.setAttribute('href', canvas.toDataURL());
                element.setAttribute('download', 'K3D-' + Date.now() + '.png');
                element.style.display = 'none';
                document.body.appendChild(element);

                element.click();

                document.body.removeChild(element);
            }, function () {
                console.error('Failed to render screenshot.');
            });
        }
    };

    gui.add(obj, 'screenshot').name('Screenshot');
}

module.exports = {
    screenshotGUI: screenshotGUI,
    getScreenshot: getScreenshot
};
