window.TestHelpers.compareCanvasWithExpectedImage =
    function (K3D, expectedImagePath, misMatchPercentage, callback) {

        'use strict';

        var header = 'data:image/png;base64,',
            xhrLoad = new XMLHttpRequest(),
            xhrSaveResult,
            xhrSaveDiff;

        xhrLoad.onreadystatechange = function () {
            if (xhrLoad.readyState === 4) {
                K3D.getScreenshot().then(function (canvas) {

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

                    var png = canvas.toDataURL();

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
                });
            }
        };

        xhrLoad.open('GET', 'http://localhost:9001/screenshots/' + expectedImagePath + '.png', true);
        xhrLoad.withCredentials = true;
        xhrLoad.send(null);
    };
