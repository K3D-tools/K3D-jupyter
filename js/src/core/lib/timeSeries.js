// jshint maxdepth:5

'use strict';

var pow10ceil = require('./helpers/math').pow10ceil,
    autoPlayed = false,
    autoPlayedHandler,
    autoPlayController,
    timeController;

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

function interpolate(a, b, f) {
    var i, interpolated;

    if (typeof (a) === 'string') {
        return (f > 0.5) ? a : b;
    }

    if (_.isNumber(a)) {
        return a + f * (b - a);
    }

    interpolated = new a.data.constructor(b.data.length);

    for (i = 0; i < interpolated.length; i++) {
        if (a.data[i] && b.data[i]) {
            interpolated[i] = a.data[i] + f * (b.data[i] - a.data[i]);
        } else {
            interpolated[i] = b.data[i];
        }
    }

    return {
        data: interpolated,
        shape: a.shape
    };
}

module.exports = {
    refreshTimeScale: function (K3D, GUI) {
        var timeSeriesInfo = getObjectsWithTimeSeriesAndMinMax(K3D);

        GUI.controls.__controllers.forEach(function (controller) {
            if (controller.property === 'time') {
                controller.min(timeSeriesInfo.min).max(timeSeriesInfo.max)
                    .step(pow10ceil(timeSeriesInfo.max - timeSeriesInfo.min) / 10000.0);

                if (timeSeriesInfo.min === timeSeriesInfo.max) {
                    controller.__li.hidden = true;
                } else {
                    controller.__li.hidden = false;
                }

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
                    interpolated_json[property] = json[property][keypoints[0].k];
                } else if (time >= keypoints[keypoints.length - 1].v) {
                    interpolated_json[property] = json[property][keypoints[keypoints.length - 1].k];
                } else {
                    for (i = 1; i < keypoints.length; i++) {
                        if (keypoints[i].v > time) {
                            a = keypoints[i - 1].v;
                            b = keypoints[i].v;
                            f = (time - a) / (b - a);

                            if (Math.abs(keypoints[i - 1].v - time) < Number.EPSILON) {
                                interpolated_json[property] = json[property][keypoints[i - 1].k];
                            } else {
                                interpolated_json[property] = interpolate(
                                    json[property][keypoints[i - 1].k],
                                    json[property][keypoints[i].k],
                                    f);
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
                if (autoPlayed) {
                    clearInterval(autoPlayedHandler);
                    autoPlayed = false;
                    autoPlayController.name('Play loop');
                } else {
                    autoPlayedHandler = setInterval(function () {
                        var t = K3D.parameters.time + 1.0 / K3D.parameters.fps;

                        if (t > timeController.__max) {
                            t = t - timeController.__max;
                        }

                        K3D.setTime(t);
                    }, 1000.0 / K3D.parameters.fps);

                    autoPlayed = true;
                    autoPlayController.name('Stop loop');
                }
            }
        };

        timeController = gui.add(K3D.parameters, 'time').min(0).max(1).name('time')
            .onChange(function (value) {
                K3D.setTime(value);
                changeParameters('time', value);
            });

        gui.add(K3D.parameters, 'fps').min(0).max(50).name('fps')
            .onChange(function (value) {
                changeParameters('fps', value);

                if (autoPlayed) {
                    obj.togglePlay();
                    obj.togglePlay();
                }
            });

        autoPlayController = gui.add(obj, 'togglePlay').name('Play loop');
    }
};
