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

function getTimeSeriesTimes(K3D) {
    const times = new Set();
    const world = K3D.getWorld();

    Object.keys(world.ObjectsListJson).forEach((id) => {
        const obj = world.ObjectsListJson[id];

        Object.keys(obj).forEach((property) => {
            if (obj[property] && typeof (obj[property].timeSeries) !== 'undefined') {
                Object.keys(obj[property]).forEach((t) => {
                    if (!Number.isNaN(parseFloat(t))) {
                        times.add(parseFloat(t));
                    }
                });
            }
        });
    });

    // The trait defaults to an empty array, whose keys would read as times 0, 1, 2...
    if (!Array.isArray(K3D.parameters.cameraAnimation)) {
        Object.keys(K3D.parameters.cameraAnimation).forEach((t) => {
            if (!Number.isNaN(parseFloat(t))) {
                times.add(parseFloat(t));
            }
        });
    }

    return Array.from(times).sort((a, b) => a - b);
}

function interpolate(a, b, f, property) {
    let i;
    let interpolated;

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
        // Frames of different size cannot be blended: the result would hold maxLength values
        // while shape still described the earlier frame. Snap to the nearer keyframe so that
        // data and shape keep describing the same thing.
        if (a.data.length !== b.data.length) {
            return (f > 0.5) ? b : a;
        }

        interpolated = new a.data.constructor(a.data.length);

        if (property === 'colors') {
            for (i = 0; i < interpolated.length; i++) {
                let r1 = (a.data[i] & 255);
                let r2 = (b.data[i] & 255);
                let g1 = ((a.data[i] >> 8) & 255);
                let g2 = ((b.data[i] >> 8) & 255);
                let b1 = ((a.data[i] >> 16) & 255);
                let b2 = ((b.data[i] >> 16) & 255);

                let rf = Math.round(r1 + f * (r2 - r1));
                let gf = Math.round(g1 + f * (g2 - g1));
                let bf = Math.round(b1 + f * (b2 - b1));

                interpolated[i] = (bf << 16) | (gf << 8) | rf;
            }
        } else {
            for (i = 0; i < interpolated.length; i++) {
                interpolated[i] = a.data[i] + f * (b.data[i] - a.data[i]);
            }
        }

        return {
            data: interpolated,
            shape: a.shape,
        };
    }

    if (a.length !== b.length) {
        return (f > 0.5) ? b : a;
    }

    interpolated = Array(a.length);

    for (i = 0; i < interpolated.length; i++) {
        interpolated[i] = a[i] + f * (b[i] - a[i]);
    }

    return interpolated;
}

function startAutoPlay(K3D, changeParameters) {
    let frameIndex = -1;

    K3D.timeSeriesStartTick = null;

    if (K3D.autoPlayed) {
        return;
    }

    K3D.autoPlayed = true;
    K3D.dispatch(K3D.events.AUTO_PLAY_CHANGE, true);

    const fallbackMaxT = getObjectsWithTimeSeriesAndMinMax(K3D).max;

    function loop(time) {
        if (!K3D.autoPlayed) {
            return;
        }

        if (K3D.timeSeriesStartTick === null) {
            K3D.timeSeriesStartTick = time - K3D.parameters.time * 1000.0 / K3D.parameters.timeSpeed;
        }

        const t = (time - K3D.timeSeriesStartTick) / 1000.0;
        const currentFrame = Math.round(t * K3D.parameters.fps);

        if (currentFrame !== frameIndex) {
            let newT = t * K3D.parameters.timeSpeed;
            const controls = K3D.GUI && K3D.GUI.controls;
            // Prefer the live GUI value so a refreshTimeScale mid-playback still applies.
            const maxT = controls ? controls.controllersMap.time._max : fallbackMaxT;

            if (newT > maxT) {
                newT -= maxT;
                K3D.timeSeriesStartTick = time - newT;
            }

            K3D.setTime(newT);
            frameIndex = currentFrame;
            changeParameters('time', newT);
        }

        requestAnimationFrame(loop);
    }

    requestAnimationFrame(loop);

    if (K3D.GUI && K3D.GUI.controls) {
        K3D.GUI.controls.controllersMap.autoPlay.name('Stop loop');
    }
}

function stopAutoPlay(K3D) {
    if (!K3D.autoPlayed) {
        return;
    }

    K3D.autoPlayed = false;
    K3D.dispatch(K3D.events.AUTO_PLAY_CHANGE, false);

    if (K3D.GUI && K3D.GUI.controls) {
        K3D.GUI.controls.controllersMap.autoPlay.name('Play loop');
    }
}

module.exports = {
    refreshTimeScale(K3D, GUI) {
        const timeSeriesInfo = getObjectsWithTimeSeriesAndMinMax(K3D);

        GUI.controls.controllersMap.time.min(timeSeriesInfo.min).max(timeSeriesInfo.max)
            .step(pow10ceil(timeSeriesInfo.max - timeSeriesInfo.min) / 10000.0);

        if (K3D.timeSeriesAnimationGUI) {
            K3D.timeSeriesAnimationGUI.domElement.hidden = timeSeriesInfo.min === timeSeriesInfo.max;
        }
    },

    interpolateTimeSeries(json, time, interpolation = true) {
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
                            if (!interpolation) {
                                interpolatedJson[property] = json[property][keypoints[i - 1].k];

                                break;
                            }

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
    getTimeSeriesTimes,

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

        const animationGUI = gui.addFolder('Animation').close();

        K3D.timeSeriesAnimationGUI = animationGUI;

        gui.controllersMap.time = animationGUI.add(K3D.parameters, 'time').min(0).max(1).name('time')
            .onChange((value) => {
                let time = value;

                // Snapped here rather than in setTime, which the autoplay loop drives with
                // continuous values.
                if (!K3D.parameters.timeInterpolation) {
                    const times = getTimeSeriesTimes(K3D);

                    if (times.length > 0) {
                        // setValue would fire onChange again; setTime's updateDisplay moves
                        // the slider instead.
                        time = times.reduce((closest, t) => (
                            Math.abs(t - value) < Math.abs(closest - value) ? t : closest
                        ), times[0]);
                    }
                }

                K3D.setTime(time);
                changeParameters('time', time);
            });

        gui.controllersMap.fps = animationGUI.add(K3D.parameters, 'fps').min(0).max(120).name('fps')
            .onChange((value) => {
                changeParameters('fps', value);
            });

        gui.controllersMap.timeSpeed = animationGUI.add(K3D.parameters, 'timeSpeed').min(0.1).max(5).step(0.01).name(
            'timeSpeed')
            .onChange((value) => {
                K3D.timeSeriesStartTick = null;
                changeParameters('timeSpeed', value);
            });

        gui.controllersMap.autoPlay = animationGUI.add(obj, 'togglePlay').name('Play loop');
    },

    startAutoPlay,
    stopAutoPlay,
};
