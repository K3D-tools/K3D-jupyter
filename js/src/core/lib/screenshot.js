'use strict';
var FileSaver = require('file-saver');
var rasterizeHTML = require('rasterizehtml');

function getScreenshot(K3D, scale) {

    // var screenshot = K3D.getWorld().renderer.domElement.toDataURL();
    // return screenshot;

    return new Promise(function (resolve, reject) {
        var finalCanvas = document.createElement('canvas'),
            finalCanvasCtx = finalCanvas.getContext('2d'),
            clearColor = K3D.parameters.clearColor.color,
            world = K3D.getWorld(),
            canvas3d = world.renderer.domElement,
            renderPromise;

        finalCanvas.width = Math.floor(canvas3d.width * scale);
        finalCanvas.height = Math.floor(canvas3d.height * scale);

        finalCanvasCtx.fillStyle = 'rgb(' +
            ((clearColor & 0xff0000) >> 16) + ',' +
            ((clearColor & 0x00ff00) >> 8) + ',' +
            (clearColor & 0x0000ff) + ')';
        finalCanvasCtx.fillRect(0, 0, finalCanvas.width, finalCanvas.height);

        renderPromise = rasterizeHTML.drawHTML('<style>body{margin:0;}</style>' +
            document.getElementById('k3d-katex').outerHTML +
            document.getElementById('k3d-style').outerHTML +
            world.overlayDOMNode.outerHTML, finalCanvas, {zoom: scale});

        renderPromise.then(function (result) {
            var arrays = world.renderOffScreen(finalCanvas.width, finalCanvas.height);

            finalCanvasCtx.scale(1, -1);
            arrays.forEach(function (array) {
                var imageData = new ImageData(array, finalCanvas.width, finalCanvas.height);
                var canvas = document.createElement('canvas');
                var ctx = canvas.getContext('2d');

                canvas.width = imageData.width;
                canvas.height = imageData.height;
                ctx.putImageData(imageData, 0, 0);

                finalCanvasCtx.drawImage(canvas, 0, 0, finalCanvas.width, -finalCanvas.height);
            });

            finalCanvasCtx.scale(1, 1);
            finalCanvasCtx.drawImage(result.image, 0, 0);

            resolve(finalCanvas);
        }, function (e) {
            reject(e);
        });
    });
}

function screenshotGUI(gui, K3D) {
    var obj = {
        screenshot: function () {
            getScreenshot(K3D, K3D.parameters.screenshotScale).then(function (canvas) {
                canvas.toBlob(function (blob) {
                    FileSaver.saveAs(blob, 'K3D-' + Date.now() + '.png');
                });
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
