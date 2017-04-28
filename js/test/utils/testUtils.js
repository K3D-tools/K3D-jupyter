window.TestHelpers.compareCanvasWithExpectedImage =
    function (canvasId, expectedImagePath, misMatchPercentage, callback) {

        'use strict';

        var header = 'data:image/png;base64,',
            finalCanvas = document.createElement('canvas'),
            canvas = document.getElementById(canvasId).getElementsByTagName('canvas')[0],
            xhrLoad = new XMLHttpRequest(),
            xhrSaveResult,
            xhrSaveDiff;

        finalCanvas.width = canvas.width;
        finalCanvas.height = canvas.height;

        xhrLoad.onreadystatechange = function () {

            var data = '<svg xmlns="http://www.w3.org/2000/svg" width="' + finalCanvas.width + '" height="' +
                finalCanvas.height + '">' +
                '<foreignObject width="100%" height="100%">' +
                document.getElementById('k3d-katex').outerHTML +
                '<div xmlns="http://www.w3.org/1999/xhtml">' +
                document.getElementById(canvasId).getElementsByTagName('div')[0].outerHTML +
                '</div>' +
                '</foreignObject>' +
                '</svg>';

            var DOMURL = window.URL || window.webkitURL || window;
            var img = new Image();
            var svg = new Blob([data], {type: 'image/svg+xml;charset=utf-8'});
            var url = DOMURL.createObjectURL(svg);

            img.onload = function () {

                var saveRender = function () {
                        var xhrSave = new XMLHttpRequest();

                        xhrSave.open('POST', 'http://localhost:9001/screenshots/' + expectedImagePath + '.png', true);
                        xhrSave.send(png.replace(header, ''));

                        return xhrSave;
                    },
                    saveDiff = function (data) {
                        var xhrSaveDiff = new XMLHttpRequest();

                        xhrSaveDiff.open('POST', 'http://localhost:9001/screenshots/' + expectedImagePath + '_diff.png',
                            true);
                        xhrSaveDiff.send(data.getImageDataUrl().replace(header, ''));

                        return xhrSaveDiff;
                    };

                if (xhrLoad.readyState === 4) {
                    finalCanvas.getContext('2d').drawImage(canvas, 0, 0);
                    finalCanvas.getContext('2d').drawImage(img, 0, 0);
                    DOMURL.revokeObjectURL(url);

                    var png = finalCanvas.toDataURL();

                    resemble.outputSettings({
                        errorColor: {
                            red: 255,
                            green: 0,
                            blue: 255
                        },
                        errorType: 'flat',
                        transparency: 0.3
                    });

                    expect(xhrLoad.status).toBe(200,
                        'Unexpected status code returned. ' + expectedImagePath + ' should exists');
                    expect(xhrLoad.response.length).toBeGreaterThan(0,
                        'Empty response received while loading reference image');

                    if (xhrLoad.status !== 200) {
                        saveRender().onreadystatechange = function () {
                            if (xhrLoad.readyState === 4) {
                                callback();
                            }
                        };
                        return;
                    }

                    resemble(png).compareTo(header + xhrLoad.response).ignoreAntialiasing().onComplete(function (data) {
                        var counter = 2;

                        function checkIfEnd(xhr) {
                            return function () {
                                if (xhr.readyState === 4) {
                                    if (--counter === 0) {
                                        callback();
                                    }
                                }
                            }
                        }

                        if (data.misMatchPercentage >= misMatchPercentage) {
                            xhrSaveDiff = saveDiff(data);
                            xhrSaveDiff.onreadystatechange = checkIfEnd(xhrSaveDiff);
                            xhrSaveResult = saveRender();
                            xhrSaveResult.onreadystatechange = checkIfEnd(xhrSaveResult);
                        }

                        expect(data.misMatchPercentage).toBeLessThan(misMatchPercentage, 'threshold mismatch');

                        if (data.misMatchPercentage < misMatchPercentage) {
                            callback();
                        }
                    });
                }
            };

            img.src = url;
        };

        xhrLoad.open('GET', 'http://localhost:9001/screenshots/' + expectedImagePath + '.png', true);
        xhrLoad.withCredentials = true;
        xhrLoad.send(null);
    };
