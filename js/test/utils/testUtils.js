'use strict';

function base64ToArrayBuffer(base64) {
    var binary_string = window.atob(base64),
        len = binary_string.length,
        bytes = new Uint8Array(len),
        i;

    for (i = 0; i < len; i++) {
        bytes[i] = binary_string.charCodeAt(i);
    }

    return new DataView(bytes.buffer);
}

function arrayToTypedArray(typedArray, array, obj) {
    // hack to preserve current samples structure

    var shape = null;

    if (typeof(obj.length) !== 'undefined') {
        shape = [obj.length, obj.height, obj.width];
    } else if (typeof(obj.width) !== 'undefined') {
        shape = [obj.height, obj.width];
    }

    if (typeof(obj.dimensions) !== 'undefined') {
        shape.push(obj.dimensions);
    }

    return {
        data: typedArray.from(array),
        shape: shape
    };
}

window.TestHelpers.fileLoader = function (url, callback) {
    var xhrLoad = new XMLHttpRequest();

    xhrLoad.open('GET', url, true);

    xhrLoad.onreadystatechange = function () {
        if (xhrLoad.readyState === 4) {
            callback(xhrLoad.response);
        }
    };

    xhrLoad.send(null);
};

window.TestHelpers.jsonLoader = function (url, callback) {
    var xhrLoad = new XMLHttpRequest(),
        json,
        converters = {
            model_matrix: arrayToTypedArray.bind(null, Float32Array),
            positions: arrayToTypedArray.bind(null, Float32Array),
            scalar_field: arrayToTypedArray.bind(null, Float32Array),
            color_map: arrayToTypedArray.bind(null, Float32Array),
            attribute: arrayToTypedArray.bind(null, Float32Array),
            vertices: arrayToTypedArray.bind(null, Float32Array),
            puv: arrayToTypedArray.bind(null, Float32Array),
            indices: arrayToTypedArray.bind(null, Uint32Array),
            colors: arrayToTypedArray.bind(null, Uint32Array),
            origins: arrayToTypedArray.bind(null, Float32Array),
            vectors: arrayToTypedArray.bind(null, Float32Array),
            heights: arrayToTypedArray.bind(null, Float32Array),
            volume: arrayToTypedArray.bind(null, Float32Array),
            voxels: arrayToTypedArray.bind(null, Uint8Array),
            sparse_voxels: arrayToTypedArray.bind(null, Uint16Array),
            binary: base64ToArrayBuffer
        };

    xhrLoad.open('GET', url, true);

    xhrLoad.onreadystatechange = function () {
        if (xhrLoad.readyState === 4) {
            json = JSON.parse(xhrLoad.response);

            json.objects.forEach(function (obj) {
                Object.keys(obj).forEach(function (key) {
                    if (typeof(converters[key]) !== 'undefined') {
                        obj[key] = converters[key](obj[key], obj);
                    }
                });
            });

            callback(json);
        }
    };

    xhrLoad.send(null);
};

window.TestHelpers.compareCanvasWithExpectedImage =
    function (K3D, expectedImagePath, misMatchPercentage, onlyCanvas, callback) {

        var header = 'data:image/png;base64,',
            xhrLoad = new XMLHttpRequest(),
            xhrSaveResult,
            xhrSaveDiff;

        xhrLoad.onreadystatechange = function () {
            if (xhrLoad.readyState === 4) {
                K3D.getScreenshot(1.0, onlyCanvas).then(function (canvas) {

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
        xhrLoad.overrideMimeType("text/plain");
        xhrLoad.send(null);
    };
