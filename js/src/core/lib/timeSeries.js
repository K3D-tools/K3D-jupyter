// jshint maxdepth:5

const THREE = require('three');
const _ = require('../../lodash');
const { pow10ceil } = require('./helpers/math');

function clone(val) {
    if (typeof (val) === 'object') {
        if (val.data) {
            return {
                data: val.data.slice(0),
                shape: val.shape,
            };
        }
        return _.cloneDeep(val);
    }

    return val;
}

function getObjectsWithTimeSeriesAndMinMax(K3D) {
    let min = 0.0;
    let max = 0.0;
    const world = K3D.getWorld();
    const objects = [];

    Object.keys(world.ObjectsListJson).forEach((id) => {
        const obj = world.ObjectsListJson[id];
        let hasTimeSeries = false;

        Object.keys(obj).forEach((property) => {
            if (obj[property] && typeof (obj[property].timeSeries) !== 'undefined') {
                hasTimeSeries = true;

                Object.keys(obj[property]).forEach((t) => {
                    if (!Number.isNaN(parseFloat(t))) {
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

    Object.keys(K3D.parameters.cameraAnimation).forEach((t) => {
        t = parseFloat(t);
        if (!Number.isNaN(t)) {
            min = Math.min(min, t);
            max = Math.max(max, t);
        }
    });

    return {
        min,
        max,
        objects,
    };
}

function interpolate(a, b, f, property) {
    let i;
    let interpolated;
    let minLength;
    let maxLength;

    if (property === 'model_matrix') {
        const matrix = new THREE.Matrix4();
        const translationA = new THREE.Vector3();
        const rotationA = new THREE.Quaternion();
        const scaleA = new THREE.Vector3();
        const translationB = new THREE.Vector3();
        const rotationB = new THREE.Quaternion();
        const scaleB = new THREE.Vector3();

        matrix.set.apply(matrix, a.data);
        matrix.decompose(translationA, rotationA, scaleA);
        matrix.set.apply(matrix, b.data);
        matrix.decompose(translationB, rotationB, scaleB);

        translationA.lerp(translationB, f);
        rotationA.slerp(rotationB, f);
        scaleA.lerp(scaleB, f);

        matrix.compose(translationA, rotationA, scaleA);
        const d = matrix.toArray();

        return {
            data: new Float32Array([
                d[0], d[4], d[8], d[12],
                d[1], d[5], d[9], d[13],
                d[2], d[6], d[10], d[14],
                d[3], d[7], d[11], d[15],
            ]),
            shape: a.shape,
        };
    }

    if (typeof (a) === 'string') {
        return (f > 0.5) ? b : a;
    }

    if (typeof (a) === 'boolean') {
        return (f > 0.5) ? b : a;
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
            shape: a.shape,
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

function startAutoPlay(K3D, changeParameters) {
    let startTick = null;
    let frameIndex = -1;

    if (K3D.autoPlayed) {
        return;
    }

    K3D.autoPlayed = true;

    function loop(time) {
        if (!K3D.autoPlayed) {
            return;
        }

        if (startTick === null) {
            startTick = time - K3D.parameters.time * 1000.0;
        }

        let t = (time - startTick) / 1000.0;
        const currentFrame = Math.round(t * K3D.parameters.fps);

        if (currentFrame !== frameIndex) {
            if (t > K3D.GUI.controls.controllersMap.time._max) {
                t -= K3D.GUI.controls.controllersMap.time._max;
                startTick = time + t;
            }

            K3D.setTime(t);
            frameIndex = currentFrame;
            changeParameters('time', t);
        }

        requestAnimationFrame(loop);
    }

    requestAnimationFrame(loop);

    K3D.GUI.controls.controllersMap.autoPlay.name('Stop loop');
}

function stopAutoPlay(K3D) {
    if (!K3D.autoPlayed) {
        return;
    }

    K3D.autoPlayed = false;
    K3D.GUI.controls.controllersMap.autoPlay.name('Play loop');
}

module.exports = {
    refreshTimeScale(K3D, GUI) {
        const timeSeriesInfo = getObjectsWithTimeSeriesAndMinMax(K3D);

        GUI.controls.controllers.forEach((controller) => {
            if (controller.property === 'time') {
                controller.min(timeSeriesInfo.min).max(timeSeriesInfo.max)
                    .step(pow10ceil(timeSeriesInfo.max - timeSeriesInfo.min) / 10000.0);
            }

            if (['togglePlay', 'fps', 'time'].indexOf(controller.property) !== -1) {
                controller.domElement.hidden = timeSeriesInfo.min === timeSeriesInfo.max;
                controller.updateDisplay();
            }
        });
    },

    interpolateTimeSeries(json, time) {
        const interpolatedJson = {};
        const changes = {};

        Object.keys(json).forEach((property) => {
            let keypoints;
            let a;
            let b;
            let i;
            let f;

            if (json[property] && typeof (json[property].timeSeries) !== 'undefined') {
                keypoints = Object.keys(json[property]).reduce((p, k) => {
                    if (!Number.isNaN(parseFloat(k))) {
                        p.push({ v: parseFloat(k), k });
                    }

                    return p;
                }, []).sort((q, w) => q.v - w.v);

                if (time <= keypoints[0].v) {
                    interpolatedJson[property] = json[property][keypoints[0].k];
                } else if (time >= keypoints[keypoints.length - 1].v) {
                    interpolatedJson[property] = json[property][keypoints[keypoints.length - 1].k];
                } else {
                    for (i = 0; i < keypoints.length; i++) {
                        if (Math.abs(keypoints[i].v - time) < 0.001) {
                            interpolatedJson[property] = clone(json[property][keypoints[i].k]);

                            break;
                        }

                        if (keypoints[i].v > time && i > 0) {
                            a = keypoints[i - 1].v;
                            b = keypoints[i].v;
                            f = (time - a) / (b - a);

                            interpolatedJson[property] = interpolate(
                                json[property][keypoints[i - 1].k],
                                json[property][keypoints[i].k],
                                f,
                                property,
                            );

                            break;
                        }
                    }
                }

                changes[property] = interpolatedJson[property];
            } else {
                interpolatedJson[property] = json[property];
            }
        });

        return { json: interpolatedJson, changes };
    },

    getObjectsWithTimeSeriesAndMinMax,

    timeSeriesGUI(gui, K3D, changeParameters) {
        const obj = {
            togglePlay() {
                if (K3D.autoPlayed) {
                    stopAutoPlay(K3D);
                } else {
                    startAutoPlay(K3D, changeParameters);
                }
            },
        };

        gui.controllersMap = gui.controllersMap || {};

        gui.controllersMap.time = gui.add(K3D.parameters, 'time').min(0).max(1).name('time')
            .onChange((value) => {
                K3D.setTime(value);
                changeParameters('time', value);
            });

        gui.controllersMap.fps = gui.add(K3D.parameters, 'fps').min(0).max(120).name('fps')
            .onChange((value) => {
                changeParameters('fps', value);
            });

        gui.controllersMap.autoPlay = gui.add(obj, 'togglePlay').name('Play loop');
    },

    startAutoPlay,
    stopAutoPlay,
};
