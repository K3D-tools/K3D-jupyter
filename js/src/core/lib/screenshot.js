'use strict';
var FileSaver = require('file-saver');
var rasterizeHTML = require('rasterizehtml');

function getScreenshot(K3D, scale, onlyCanvas) {

    return new Promise(function (resolve, reject) {
        var finalCanvas = document.createElement('canvas'),
            finalCanvasCtx = finalCanvas.getContext('2d'),
            htmlElementCanvas = document.createElement('canvas'),
            clearColor = K3D.parameters.clearColor,
            world = K3D.getWorld(),
            canvas3d = world.renderer.domElement,
            renderPromise,
            t;

        t = new Date().getTime();
        finalCanvas.width = htmlElementCanvas.width = Math.floor(canvas3d.width * scale);
        finalCanvas.height = htmlElementCanvas.height = Math.floor(canvas3d.height * scale);

        K3D.labels = [];
        K3D.dispatch(K3D.events.BEFORE_RENDER);

        if (clearColor >= 0) {
            finalCanvasCtx.fillStyle = 'rgb(' +
                                       ((clearColor & 0xff0000) >> 16) + ',' +
                                       ((clearColor & 0x00ff00) >> 8) + ',' +
                                       (clearColor & 0x0000ff) + ')';

            finalCanvasCtx.fillRect(0, 0, finalCanvas.width, finalCanvas.height);
        }

        if (onlyCanvas) {
            renderPromise = Promise.resolve();
        } else {
            renderPromise = rasterizeHTML.drawHTML(
                '<style>body{margin:0;}</style>' +
                document.getElementById('k3d-katex').outerHTML +
                document.getElementById('k3d-style').outerHTML +
                world.overlayDOMNode.outerHTML +
                (K3D.colorMapNode ? K3D.colorMapNode.outerHTML : ''),
                htmlElementCanvas,
                {
                    zoom: scale
                });
        }

        renderPromise.then(function (result) {
            console.log('K3D: Screenshot [html]: ' + (new Date().getTime() - t) / 1000, 's');
            t = new Date().getTime();

            world.renderOffScreen(finalCanvas.width, finalCanvas.height).then(function (arrays) {
                console.log('K3D: Screenshot [canvas]: ' + (new Date().getTime() - t) / 1000, 's');
                t = new Date().getTime();

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

                if (result) {
                    finalCanvasCtx.scale(1, -1);
                    finalCanvasCtx.drawImage(htmlElementCanvas, 0, 0);
                }

                resolve(finalCanvas);
            });
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
                    var filename = 'K3D-' + Date.now() + '.png';

                    if (K3D.parameters.name) {
                        filename = K3D.parameters.name + '.png';
                    }

                    FileSaver.saveAs(blob, filename);
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
