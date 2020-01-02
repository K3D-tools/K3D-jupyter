// jshint maxdepth:5

'use strict';

var THREE = require('three'),
    pow10ceil = require('./helpers/math').pow10ceil;

function getObjectsWithTimeSeriesAndMinMax(K3D) {
    var min = 0.0, max = 0.0,
        world = K3D.getWorld(),
        objects = [];

    Object.keys(world.ObjectsListJson).forEach(function (id) {
        var obj = world.ObjectsListJson[id],
            hasTimeSeries = false;

        Object.keys(obj).forEach(function (property) {
            if (obj[property] && typeof (obj[property].timeSeries) !== 'undefined') {
                hasTimeSeries = true;

                Object.keys(obj[property]).forEach(function (t) {
                    if (!isNaN(parseFloat(t))) {
                        min = Math.min(min, parseFloat(t));
                        max = Math.max(max, parseFloat(t));
                    }
                });
            }
        });

        if (hasTimeSeries) {
            objects.push(obj);
        }
    });

    return {
        min: min,
        max: max,
        objects: objects
    };
}

function interpolate(a, b, f, property) {
    var i, interpolated, minLength, maxLength;

    if (property === 'model_matrix') {
        var matrix = new THREE.Matrix4(),
            translationA = new THREE.Vector3(),
            rotationA = new THREE.Quaternion(),
            scaleA = new THREE.Vector3(),
            translationB = new THREE.Vector3(),
            rotationB = new THREE.Quaternion(),
            scaleB = new THREE.Vector3(),
            d;

        matrix.set.apply(matrix, a.data);
        matrix.decompose(translationA, rotationA, scaleA);
        matrix.set.apply(matrix, b.data);
        matrix.decompose(translationB, rotationB, scaleB);

        translationA.lerp(translationB, f);
        rotationA.slerp(rotationB, f);
        scaleA.lerp(scaleB, f);

        matrix.compose(translationA, rotationA, scaleA);
        d = matrix.toArray();

        return {
            data: new Float32Array([
                d[0], d[4], d[8], d[12],
                d[1], d[5], d[9], d[13],
                d[2], d[6], d[10], d[14],
                d[3], d[7], d[11], d[15]
            ]),
            shape: a.shape
        };
    }

    if (typeof (a) === 'string') {
        return (f > 0.5) ? a : b;
    }

    if (_.isNumber(a)) {
        return a + f * (b - a);
    }

    if (a.data) {
        minLength = Math.min(a.data.length, b.data.length);
        maxLength = Math.max(a.data.length, b.data.length);
        interpolated = new a.data.constructor(maxLength);

        for (i = 0; i < minLength; i++) {
            interpolated[i] = a.data[i] + f * (b.data[i] - a.data[i]);
        }

        if (minLength !== maxLength) {
            for (i = minLength; i < maxLength; i++) {
                interpolated[i] = a.data[i] || b.data[i];
            }
        }

        return {
            data: interpolated,
            shape: a.shape
        };
    }

    minLength = Math.min(a.length, b.length);
    maxLength = Math.max(a.length, b.length);
    interpolated = Array(maxLength);

    for (i = 0; i < interpolated.length; i++) {
        interpolated[i] = a[i] + f * (b[i] - a[i]);
    }

    if (minLength !== maxLength) {
        for (i = minLength; i < maxLength; i++) {
            interpolated[i] = a[i] || b[i];
        }
    }

    return interpolated;
}

function startAutoPlay(K3D) {
    if (K3D.autoPlayedHandler) {
        return;
    }

    K3D.autoPlayedFps = K3D.parameters.fps;

    K3D.autoPlayedHandler = setInterval(function () {
        if (K3D.autoPlayedFps !== K3D.parameters.fps) {
            clearInterval(K3D.autoPlayedHandler);
            K3D.autoPlayedHandler = false;
            startAutoPlay(K3D);

            return;
        }

        var t = K3D.parameters.time + 1.0 / K3D.parameters.fps;

        if (t > K3D.GUI.controls.controllersMap.time.__max) {
            t = t - K3D.GUI.controls.controllersMap.time.__max;
        }

        K3D.setTime(t);
    }, 1000.0 / K3D.parameters.fps);

    K3D.GUI.controls.controllersMap.autoPlay.name('Stop loop');
}

function stopAutoPlay(K3D) {
    if (!K3D.autoPlayedHandler) {
        return;
    }

    clearInterval(K3D.autoPlayedHandler);
    K3D.autoPlayedHandler = false;
    K3D.GUI.controls.controllersMap.autoPlay.name('Play loop');
}

module.exports = {
    refreshTimeScale: function (K3D, GUI) {
        var timeSeriesInfo = getObjectsWithTimeSeriesAndMinMax(K3D);

        GUI.controls.__controllers.forEach(function (controller) {
            if (controller.property === 'time') {
                controller.min(timeSeriesInfo.min).max(timeSeriesInfo.max)
                    .step(pow10ceil(timeSeriesInfo.max - timeSeriesInfo.min) / 10000.0);
            }

            if (['togglePlay', 'fps', 'time'].indexOf(controller.property) !== -1) {
                controller.__li.hidden = timeSeriesInfo.min === timeSeriesInfo.max;
                controller.updateDisplay();
            }
        });
    },

    interpolateTimeSeries: function (json, time) {
        var interpolated_json = {}, changes = {};

        Object.keys(json).forEach(function (property) {
            var keypoints,
                a, b, i, f;

            if (json[property] && typeof (json[property].timeSeries) !== 'undefined') {
                keypoints = Object.keys(json[property]).reduce(function (p, k) {
                    if (!isNaN(parseFloat(k))) {
                        p.push({v: parseFloat(k), k: k});
                    }

                    return p;
                }, []).sort(function (a, b) {
                    return a.v - b.v;
                });

                if (time <= keypoints[0].v) {
                    interpolated_json[property] = _.cloneDeep(json[property][keypoints[0].k]);
                } else if (time >= keypoints[keypoints.length - 1].v) {
                    interpolated_json[property] = _.cloneDeep(json[property][keypoints[keypoints.length - 1].k]);
                } else {
                    for (i = 1; i < keypoints.length; i++) {
                        if (keypoints[i].v > time) {
                            if (Math.abs(keypoints[i - 1].v - time) < Number.EPSILON) {
                                interpolated_json[property] = _.cloneDeep(json[property][keypoints[i - 1].k]);
                            } else {
                                a = keypoints[i - 1].v;
                                b = keypoints[i].v;
                                f = (time - a) / (b - a);

                                interpolated_json[property] = interpolate(
                                    json[property][keypoints[i - 1].k],
                                    json[property][keypoints[i].k],
                                    f,
                                    property);
                            }

                            break;
                        }
                    }
                }

                changes[property] = interpolated_json[property];
            } else {
                interpolated_json[property] = json[property];
            }
        });

        return {json: interpolated_json, changes: changes};
    },

    getObjectsWithTimeSeriesAndMinMax: getObjectsWithTimeSeriesAndMinMax,

    timeSeriesGUI: function (gui, K3D, changeParameters) {
        var obj = {
            togglePlay: function () {
                if (K3D.autoPlayedHandler) {
                    stopAutoPlay(K3D);
                } else {
                    startAutoPlay(K3D);
                }
            }
        };

        gui.controllersMap = gui.controllersMap || {};

        gui.controllersMap.time = gui.add(K3D.parameters, 'time').min(0).max(1).name('time')
            .onChange(function (value) {
                K3D.setTime(value);
                changeParameters('time', value);
            });

        gui.controllersMap.fps = gui.add(K3D.parameters, 'fps').min(0).max(120).name('fps')
            .onChange(function (value) {
                changeParameters('fps', value);
            });

        gui.controllersMap.autoPlay = gui.add(obj, 'togglePlay').name('Play loop');
    },

    startAutoPlay: startAutoPlay,
    stopAutoPlay: stopAutoPlay
};
