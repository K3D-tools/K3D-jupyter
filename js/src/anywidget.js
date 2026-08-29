// anywidget front-end module (AFM) for K3D. One ESM serves all widget classes;
// the python side stamps _kind on every class and the hooks dispatch on it.
import 'katex/dist/katex.min.css';
import * as fflate from 'fflate';
import _ from './lodash';
import K3D from './core/Core';
import K3DTransferFunctionEditor from './core/lib/transferFunctionEditorCore';
import serialize from './core/lib/helpers/serialize';
import buffer from './core/lib/helpers/buffer';
import msgpack from './core/lib/helpers/msgpackCodec';
import ThreeJsProvider from './providers/threejs/provider';
import { viewModes } from './core/lib/viewMode';
import bvhWorkerSource from './core/lib/bvhWorkerSource';

// the module can be instantiated more than once (one _esm per widget class), and the
// object/chunk stub modules may have created the registry first, so every field is
// ensured rather than assigned
globalThis.__k3dWidgets = globalThis.__k3dWidgets || {};
const REG = globalThis.__k3dWidgets;

REG.objects = REG.objects || {};
REG.chunks = REG.chunks || {};
REG.plots = REG.plots || [];
REG.pending = REG.pending || [];

function runOnEveryPlot(id, cb) {
    REG.plots.forEach((plot) => {
        if (plot.model.get('object_ids').indexOf(id) !== -1) {
            cb(plot, plot.K3DInstance.getObjectById(id));
        }
    });
}

function deserialized(model, key) {
    return serialize.deserialize(model.get(key));
}

// This module is imported from a blob URL, so nothing next to it has a resolvable URL: its
// worker chunk comes from the kernel that served the module. Resolves null on anything going
// wrong, including a kernel that never answers - the caller then does the work itself.
function fetchWidgetAsset(model, name) {
    return new Promise((resolve) => {
        function onMessage(msg, buffers) {
            if (!msg || msg.msg_type !== 'widget_asset' || msg.name !== name) {
                return;
            }

            model.off('msg:custom', onMessage);

            if (!buffers || buffers.length === 0) {
                resolve(null);

                return;
            }

            const raw = buffers[0];
            const bytes = raw instanceof Uint8Array ? raw : new Uint8Array(raw.buffer || raw);

            resolve(new TextDecoder().decode(bytes));
        }

        model.on('msg:custom', onMessage);
        model.send({ msg_type: 'fetch_widget_asset', name });

        setTimeout(() => {
            model.off('msg:custom', onMessage);
            resolve(null);
        }, 10000);
    });
}

/* ------------------------------------------------------------------------- */
/* object relay                                                               */
/*                                                                            */
/* Colab (and any frontend that materialises widget models lazily) renders    */
/* each output in its own context: the plot model exists there, but the       */
/* object models never do - object_ids are plain integers, not model          */
/* references. The plot's own comm relays their state instead, in the .k3d    */
/* binary snapshot encoding (zlib over msgpack).                              */
/* ------------------------------------------------------------------------- */

function relayU8(buffers) {
    const b = buffers[0];

    return new Uint8Array(b.buffer, b.byteOffset || 0, b.byteLength);
}

function registerRelayedObject(dict) {
    const attrs = {};

    Object.keys(dict).forEach((k) => {
        attrs[k] = serialize.deserialize(dict[k]);
    });

    const existing = REG.objects[attrs.id];

    // a live model is authoritative - the relay only fills contexts without one
    if (existing && existing.model) {
        return existing.attributes;
    }

    REG.objects[attrs.id] = { model: null, attributes: attrs };

    return attrs;
}

function registerRelayedChunk(dict) {
    const attrs = {};

    Object.keys(dict).forEach((k) => {
        attrs[k] = serialize.deserialize(dict[k]);
    });

    REG.chunks[attrs.id] = { attributes: attrs };
}

function requestMissingObjects(view, ids) {
    const missing = ids.filter((id) => !REG.objects[id] && !view.pendingFetch.has(id));

    if (missing.length === 0) {
        return;
    }

    missing.forEach((id) => view.pendingFetch.add(id));
    view.model.send({ msg_type: 'fetch_objects', ids: missing });
}

// typed arrays ride to the kernel as plain msgpack bin + dtype, so the python
// side can read them with from_json instead of a custom ext codec
function relayEncodableValue(value) {
    if (value && value.data && value.data.buffer) {
        return {
            ...value,
            data: new Uint8Array(value.data.buffer, value.data.byteOffset, value.data.byteLength),
        };
    }

    return value;
}

// deserialized snapshot of every synced trait - the equivalent of the old
// Backbone model.attributes that K3DInstance.load/reload consumes
function attrsOf(model) {
    const attrs = {};

    model.get('_synced_props').forEach((key) => {
        attrs[key] = deserialized(model, key);
    });

    return attrs;
}

function saveChanges(model, key, value) {
    // model.set fires 'change:' listeners synchronously - the bridges below must not
    // reload the scene for a change the scene itself just produced (a second reload
    // in the same tick double-creates the object and orphans one copy)
    model._k3dOwnChange = (model._k3dOwnChange || 0) + 1;

    // the comm can be gone (kernel restart, page teardown) - a failed echo must not throw
    try {
        model.set(key, value);
        model.save_changes();
    } catch (e) {
        console.log(e);
    } finally {
        model._k3dOwnChange -= 1;
    }
}

// K3D needs a laid-out node; anywidget can call render before el joins the document
function whenConnected(el) {
    return new Promise((resolve) => {
        (function check() {
            if (el.isConnected) {
                resolve();
            } else {
                requestAnimationFrame(check);
            }
        }());
    });
}

/* ------------------------------------------------------------------------- */
/* objects                                                                    */
/* ------------------------------------------------------------------------- */

function initObject({ model }) {
    const attrs = attrsOf(model);

    REG.objects[attrs.id] = { model, attributes: attrs };

    model.get('_synced_props').forEach((key) => {
        model.on(`change:${key}`, () => {
            if (model._k3dOwnChange) {
                return;
            }

            const value = deserialized(model, key);

            attrs[key] = value;

            const changed = {};

            changed[key] = value;

            REG.plots.forEach((plot) => {
                plot.refreshObject(attrs.id, changed);
            });
        });
    });

    model.on('msg:custom', (msg) => {
        if (msg.msg_type === 'fetch') {
            let property = attrs[msg.field];

            if (property && typeof (property.data) !== 'undefined'
                && typeof (property.shape) !== 'undefined') {
                property = serialize.serialize({
                    data: property.data,
                    shape: property.shape,
                    compression_level: attrs.compression_level,
                });
            } else if (_.isObject(property) && property !== null) {
                // force change detection for plain objects echoed back unchanged
                property = { ...property, _lastModified: Date.now() };
            }

            saveChanges(model, msg.field, property);
        }

        if (msg.msg_type === 'shadow_map_update' && attrs.type === 'Volume') {
            runOnEveryPlot(attrs.id, (plot, objInstance) => {
                if (objInstance && objInstance.refreshLightMap) {
                    objInstance.refreshLightMap(msg.direction);
                    plot.K3DInstance.render();
                }
            });
        }
    });

    return () => {
        delete REG.objects[attrs.id];
    };
}

/* ------------------------------------------------------------------------- */
/* voxel chunks                                                               */
/* ------------------------------------------------------------------------- */

function initChunk({ model }) {
    const attrs = attrsOf(model);

    REG.chunks[attrs.id] = { attributes: attrs };

    model.get('_synced_props').forEach((key) => {
        model.on(`change:${key}`, () => {
            if (model._k3dOwnChange) {
                return;
            }

            attrs[key] = deserialized(model, key);

            Object.keys(REG.objects).forEach((id) => {
                if (REG.objects[id].attributes.type === 'VoxelsGroup') {
                    runOnEveryPlot(REG.objects[id].attributes.id, (plot, objInstance) => {
                        objInstance.updateChunk(attrs);
                    });
                }
            });
        });
    });

    return () => {
        delete REG.chunks[attrs.id];
    };
}

/* ------------------------------------------------------------------------- */
/* plot                                                                       */
/* ------------------------------------------------------------------------- */

function deserializedEnvironment(model) {
    const env = model.get('environment');

    if (env !== null && typeof (env) === 'object') {
        const result = serialize.deserialize(env);

        // the catalog name rides along for the GUI
        if (env.name) {
            result.name = env.name;
        }

        return result;
    }

    return env;
}

// every handler is a 1:1 bridge trait -> K3DInstance setter, same as the old PlotView
const PLOT_HANDLERS = {
    camera_auto_fit: (v) => v.K3DInstance.setCameraAutoFit(v.model.get('camera_auto_fit')),
    lighting: (v) => v.K3DInstance.setDirectionalLightingIntensity(v.model.get('lighting')),
    time: (v) => {
        if (v.K3DInstance.parameters.time !== v.model.get('time')) {
            v.renderPromises.push(v.K3DInstance.setTime(v.model.get('time')));
        }
    },
    fps: (v) => v.K3DInstance.setFps(v.model.get('fps')),
    time_speed: (v) => v.K3DInstance.setTimeSpeed(v.model.get('time_speed')),
    time_interpolation: (v) => v.K3DInstance.setTimeInterpolation(v.model.get('time_interpolation')),
    grid_auto_fit: (v) => v.K3DInstance.setGridAutoFit(v.model.get('grid_auto_fit')),
    grid_visible: (v) => v.K3DInstance.setGridVisible(v.model.get('grid_visible')),
    grid_color: (v) => v.K3DInstance.setGridColor(v.model.get('grid_color')),
    label_color: (v) => v.K3DInstance.setLabelColor(v.model.get('label_color')),
    depth_peels: (v) => v.K3DInstance.setDepthPeels(v.model.get('depth_peels')),
    renderer: (v) => v.K3DInstance.setRenderer(v.model.get('renderer')),
    environment: (v) => v.K3DInstance.setEnvironment(deserializedEnvironment(v.model)),
    environment_rotation: (v) => v.K3DInstance.setEnvironmentRotation(v.model.get('environment_rotation')),
    tone_mapping: (v) => v.K3DInstance.setToneMapping(v.model.get('tone_mapping')),
    ao_radius: (v) => v.K3DInstance.setAORadius(v.model.get('ao_radius')),
    ao_strength: (v) => v.K3DInstance.setAOStrength(v.model.get('ao_strength')),
    cinematic_samples: (v) => v.K3DInstance.setCinematicSamples(v.model.get('cinematic_samples')),
    cinematic_bounces: (v) => v.K3DInstance.setCinematicBounces(v.model.get('cinematic_bounces')),
    cinematic_glossy_filter: (v) => v.K3DInstance.setCinematicGlossyFilter(v.model.get('cinematic_glossy_filter')),
    fps_meter: (v) => v.K3DInstance.setFpsMeter(v.model.get('fps_meter')),
    screenshot_scale: (v) => v.K3DInstance.setScreenshotScale(v.model.get('screenshot_scale')),
    voxel_paint_color: (v) => v.K3DInstance.setVoxelPaint(v.model.get('voxel_paint_color')),
    background_color: (v) => v.K3DInstance.setClearColor(v.model.get('background_color')),
    grid: (v) => v.K3DInstance.setGrid(v.model.get('grid')),
    render_on_change: (v) => v.K3DInstance.setRenderOnChange(v.model.get('render_on_change')),
    camera: (v) => v.K3DInstance.setCamera(v.model.get('camera')),
    camera_animation: (v) => v.K3DInstance.setCameraAnimation(v.model.get('camera_animation')),
    clipping_planes: (v) => v.K3DInstance.setClippingPlanes(v.model.get('clipping_planes')),
    slice_viewer_mask_object_ids: (v) => v.K3DInstance.setSliceViewerMaskObjects(
        v.model.get('slice_viewer_mask_object_ids'),
    ),
    object_ids: (v) => {
        const previous = v.objectIds;
        const current = v.model.get('object_ids');

        v.objectIds = current;

        _.difference(previous, current).forEach((id) => {
            v.renderPromises.push(v.K3DInstance.removeObject(id));
        });

        const added = _.difference(current, previous);

        added.forEach((id) => {
            if (REG.objects[id]) {
                v.renderPromises.push(v.K3DInstance.load({ objects: [REG.objects[id].attributes] }));
            }
        });

        // models absent in this context (Colab renders outputs in isolated frames)
        // arrive over the plot comm as an objects_state message
        requestMissingObjects(v, added);
    },
    menu_visibility: (v) => v.K3DInstance.setMenuVisibility(v.model.get('menu_visibility')),
    colorbar_object_id: (v) => v.K3DInstance.setColorMapLegend(v.model.get('colorbar_object_id')),
    slice_viewer_object_id: (v) => v.K3DInstance.setSliceViewer(v.model.get('slice_viewer_object_id')),
    slice_viewer_direction: (v) => v.K3DInstance.setSliceViewerDirection(v.model.get('slice_viewer_direction')),
    colorbar_scientific: (v) => v.K3DInstance.setColorbarScientific(v.model.get('colorbar_scientific')),
    rendering_steps: (v) => v.K3DInstance.setRenderingSteps(v.model.get('rendering_steps')),
    axes: (v) => v.K3DInstance.setAxes(v.model.get('axes')),
    camera_no_rotate: (v) => PLOT_HANDLERS._cameraLock(v),
    camera_no_zoom: (v) => PLOT_HANDLERS._cameraLock(v),
    camera_no_pan: (v) => PLOT_HANDLERS._cameraLock(v),
    _cameraLock: (v) => v.K3DInstance.setCameraLock(
        v.model.get('camera_no_rotate'),
        v.model.get('camera_no_zoom'),
        v.model.get('camera_no_pan'),
    ),
    camera_rotate_speed: (v) => PLOT_HANDLERS._cameraSpeeds(v),
    camera_zoom_speed: (v) => PLOT_HANDLERS._cameraSpeeds(v),
    camera_pan_speed: (v) => PLOT_HANDLERS._cameraSpeeds(v),
    _cameraSpeeds: (v) => v.K3DInstance.setCameraSpeeds(
        v.model.get('camera_rotate_speed'),
        v.model.get('camera_zoom_speed'),
        v.model.get('camera_pan_speed'),
    ),
    camera_fov: (v) => v.K3DInstance.setCameraFOV(v.model.get('camera_fov')),
    camera_damping_factor: (v) => v.K3DInstance.setCameraDampingFactor(v.model.get('camera_damping_factor')),
    camera_up_axis: (v) => v.K3DInstance.setCameraUpAxis(v.model.get('camera_up_axis')),
    axes_helper: (v) => v.K3DInstance.setAxesHelper(v.model.get('axes_helper')),
    axes_helper_colors: (v) => v.K3DInstance.setAxesHelperColors(v.model.get('axes_helper_colors')),
    snapshot_type: (v) => v.K3DInstance.setSnapshotType(v.model.get('snapshot_type')),
    name: (v) => v.K3DInstance.setName(v.model.get('name')),
    mode: (v) => v.K3DInstance.setViewMode(v.model.get('mode')),
    minimum_fps: (v) => v.K3DInstance.setMinimumFps(v.model.get('minimum_fps')),
    camera_mode: (v) => v.K3DInstance.setCameraMode(v.model.get('camera_mode')),
    manipulate_mode: (v) => v.K3DInstance.setManipulateMode(v.model.get('manipulate_mode')),
    hidden_object_ids: (v) => v.K3DInstance.setHiddenObjectIds(v.model.get('hidden_object_ids')),
    additional_js_code: (v) => {
        v.K3DInstance.setAdditionalJsCode(v.model.get('additional_js_code'));
        v.evalInContext();
    },
};

function renderPlot({ model, el }) {
    const containerEnvelope = window.document.createElement('div');
    const container = window.document.createElement('div');

    bvhWorkerSource.provide(() => fetchWidgetAsset(model, 'k3d-bvh-worker.mjs'));

    containerEnvelope.style.cssText = [
        `height:${model.get('height')}px`,
        'position: relative',
    ].join(';');

    container.style.cssText = [
        'width: 100%',
        'height: 100%',
        'position: relative',
    ].join(';');

    containerEnvelope.appendChild(container);
    el.classList.add('k3d-plot');
    el.appendChild(containerEnvelope);

    const view = {
        model,
        K3DInstance: null,
        renderPromises: [],
        objectIds: model.get('object_ids'),
        pendingFetch: new Set(),
        lastCameraSync: Date.now(),
        cameraSyncTimeout: null,
        pendingCamera: null,
        evalInContext() {
            const K3DInstance = view.K3DInstance;
            // eslint-disable-next-line no-eval
            eval(K3DInstance.parameters.additionalJsCode);
        },
        refreshObject(id, changed) {
            if (model.get('object_ids').indexOf(id) !== -1) {
                view.renderPromises.push(view.K3DInstance.reload(REG.objects[id].attributes, changed));
            }
        },
    };

    let disposed = false;
    const resizeObserver = new ResizeObserver(() => {
        if (view.K3DInstance) {
            view.K3DInstance.resizeHelper();
        }
    });
    const contextMenuListener = (event) => {
        if (container.contains(event.target)) {
            event.preventDefault();
            event.stopPropagation();
        }
    };

    whenConnected(el).then(() => {
        if (disposed) {
            return;
        }

        REG.plots.push(view);

        model.on('msg:custom', (obj, buffers) => {
            if (obj.msg_type === 'snapshot_source' && buffers && buffers.length > 0) {
                window.k3dCompressed = buffer.arrayBufferToBase64(buffers[0].buffer);
            }

            if (obj.msg_type === 'objects_state' && buffers && buffers.length > 0) {
                const state = msgpack.decode(fflate.unzlibSync(relayU8(buffers)));

                (state.chunkList || []).forEach(registerRelayedChunk);

                (state.objects || []).forEach((dict) => {
                    const attrs = registerRelayedObject(dict);

                    view.pendingFetch.delete(attrs.id);

                    if (model.get('object_ids').indexOf(attrs.id) !== -1
                        && !view.K3DInstance.getObjectById(attrs.id)) {
                        view.renderPromises.push(view.K3DInstance.load({ objects: [attrs] }));
                    }
                });
            }

            if (obj.msg_type === 'object_patch' && buffers && buffers.length > 0) {
                const patch = msgpack.decode(fflate.unzlibSync(relayU8(buffers)));
                const entry = REG.objects[patch.id];

                // with a live model in this context the trait sync delivers the same
                // change - applying the relay copy too would reload twice
                if (entry && entry.model === null) {
                    const value = serialize.deserialize(patch.value);

                    entry.attributes[patch.key] = value;

                    const changed = {};

                    changed[patch.key] = value;

                    REG.plots.forEach((plot) => {
                        plot.refreshObject(patch.id, changed);
                    });
                }
            }

            if (obj.msg_type === 'fetch_screenshot') {
                view.K3DInstance.getScreenshot(view.K3DInstance.parameters.screenshotScale, obj.only_canvas)
                    .then((canvas) => {
                        saveChanges(model, 'screenshot', canvas.toDataURL().split(',')[1]);
                    });
            }

            if (obj.msg_type === 'fetch_snapshot') {
                saveChanges(model, 'snapshot', view.K3DInstance.getHTMLSnapshot(obj.compression_level));
            }

            if (obj.msg_type === 'fetch_gltf') {
                view.K3DInstance.getGLTF().then((glb) => {
                    saveChanges(model, 'gltf', buffer.arrayBufferToBase64(glb));
                }, (e) => {
                    console.error('Failed to export glTF.', e);
                });
            }

            if (obj.msg_type === 'start_auto_play') {
                view.K3DInstance.startAutoPlay();
            }

            if (obj.msg_type === 'stop_auto_play') {
                view.K3DInstance.stopAutoPlay();
            }

            if (obj.msg_type === 'reset_camera') {
                view.K3DInstance.resetCamera(obj.factor);
            }

            if (obj.msg_type === 'render') {
                if (view.renderPromises.length === 0) {
                    view.K3DInstance.refreshAfterObjectsChange(false, true);
                } else {
                    Promise.all(view.renderPromises).then((values) => {
                        view.K3DInstance.refreshAfterObjectsChange(false, true);

                        if (values.length === view.renderPromises.length) {
                            view.renderPromises = [];
                        }
                    });
                }
            }
        });

        Object.keys(PLOT_HANDLERS).forEach((key) => {
            if (key.charAt(0) === '_') {
                return;
            }
            model.on(`change:${key}`, () => {
                if (model._k3dOwnChange) {
                    return;
                }

                PLOT_HANDLERS[key](view);
            });
        });

        try {
            view.K3DInstance = new K3D(ThreeJsProvider, container, {
                antialias: model.get('antialias'),
                logarithmicDepthBuffer: model.get('logarithmic_depth_buffer'),
                lighting: model.get('lighting'),
                cameraMode: model.get('camera_mode'),
                snapshotType: model.get('snapshot_type'),
                backendVersion: model.get('_backend_version'),
                screenshotScale: model.get('screenshot_scale'),
                menuVisibility: model.get('menu_visibility'),
                cameraAutoFit: model.get('camera_auto_fit'),
                cameraNoRotate: model.get('camera_no_rotate'),
                cameraNoZoom: model.get('camera_no_zoom'),
                cameraNoPan: model.get('camera_no_pan'),
                cameraRotateSpeed: model.get('camera_rotate_speed'),
                cameraZoomSpeed: model.get('camera_zoom_speed'),
                cameraPanSpeed: model.get('camera_pan_speed'),
                cameraDampingFactor: model.get('camera_damping_factor'),
                cameraFov: model.get('camera_fov'),
                colorbarObjectId: model.get('colorbar_object_id'),
                cameraAnimation: model.get('camera_animation'),
                sliceViewerMaskObjectIds: model.get('slice_viewer_mask_object_ids'),
                sliceViewerObjectId: model.get('slice_viewer_object_id'),
                sliceViewerDirection: model.get('slice_viewer_direction'),
                name: model.get('name'),
                axes: model.get('axes'),
                axesHelper: model.get('axes_helper'),
                grid: model.get('grid'),
                fps: model.get('fps'),
                timeInterpolation: model.get('time_interpolation'),
                depthPeels: model.get('depth_peels'),
                renderer: model.get('renderer'),
                environment: deserializedEnvironment(model),
                environmentRotation: model.get('environment_rotation'),
                toneMapping: model.get('tone_mapping'),
                aoRadius: model.get('ao_radius'),
                aoStrength: model.get('ao_strength'),
                cinematicSamples: model.get('cinematic_samples'),
                cinematicBounces: model.get('cinematic_bounces'),
                cinematicGlossyFilter: model.get('cinematic_glossy_filter'),
                renderOnChange: model.get('render_on_change'),
                gridVisible: model.get('grid_visible'),
                gridColor: model.get('grid_color'),
                gridAutoFit: model.get('grid_auto_fit'),
                clippingPlanes: model.get('clipping_planes'),
                labelColor: model.get('label_color'),
                voxelPaintColor: model.get('voxel_paint_color'),
                hiddenObjectIds: model.get('hidden_object_ids'),
                additionalJsCode: model.get('additional_js_code'),
            });

            if (model.get('camera_auto_fit') === false) {
                view.K3DInstance.setCamera(model.get('camera'));
            }
        } catch (e) {
            console.log(e);
            return;
        }

        view.K3DInstance.setClearColor(model.get('background_color'));
        view.K3DInstance.setChunkList(REG.chunks);

        model.get('object_ids').forEach((id) => {
            if (REG.objects[id]) {
                view.renderPromises.push(view.K3DInstance.load({ objects: [REG.objects[id].attributes] }));
            }
        });

        requestMissingObjects(view, model.get('object_ids'));

        view.cameraChangeId = view.K3DInstance.on(view.K3DInstance.events.CAMERA_CHANGE, (control) => {
            const now = Date.now();
            const sinceLast = now - view.lastCameraSync;

            if (sinceLast > 200) {
                view.lastCameraSync = now;
                view.pendingCamera = null;
                saveChanges(model, 'camera', control);
                return;
            }

            view.pendingCamera = control;

            if (view.cameraSyncTimeout === null) {
                view.cameraSyncTimeout = setTimeout(() => {
                    view.cameraSyncTimeout = null;

                    if (view.pendingCamera) {
                        view.lastCameraSync = Date.now();
                        saveChanges(model, 'camera', view.pendingCamera);
                        view.pendingCamera = null;
                    }
                }, 200 - sinceLast);
            }
        });

        view.GUIObjectChanges = view.K3DInstance.on(view.K3DInstance.events.OBJECT_CHANGE, (change) => {
            const entry = REG.objects[change.id];

            if (!entry) {
                return;
            }

            let { value } = change;

            if (value && value.data && value.shape) {
                value = serialize.serialize({
                    data: value.data,
                    shape: value.shape,
                    compression_level: entry.attributes.compression_level,
                });
            }

            // relayed objects have no model in this context - route the edit
            // through the plot comm instead
            if (entry.model === null) {
                entry.attributes[change.key] = change.value;
                view.model.send({ msg_type: 'object_change' }, undefined, [
                    fflate.zlibSync(msgpack.encode({
                        id: change.id,
                        key: change.key,
                        value: relayEncodableValue(value),
                    })),
                ]);
                return;
            }

            saveChanges(entry.model, change.key, value);
        });

        view.GUIParametersChanges = view.K3DInstance.on(
            view.K3DInstance.events.PARAMETERS_CHANGE,
            (change) => saveChanges(model, change.key, change.value),
        );

        view.voxelsCallback = view.K3DInstance.on(view.K3DInstance.events.VOXELS_CALLBACK, (param) => {
            const entry = REG.objects[param.object.K3DIdentifier];

            if (entry && entry.model) {
                entry.model.send({
                    msg_type: 'click_callback',
                    coord: param.coord,
                });
            }
        });

        view.objectHoverCallback = view.K3DInstance.on(view.K3DInstance.events.OBJECT_HOVERED, (param) => {
            const entry = REG.objects[param.K3DIdentifier];

            if (entry && entry.model && view.K3DInstance.parameters.viewMode === viewModes.callback) {
                entry.model.send(_.extend({ msg_type: 'hover_callback' }, param));
            }
        });

        view.objectClickCallback = view.K3DInstance.on(view.K3DInstance.events.OBJECT_CLICKED, (param) => {
            const entry = REG.objects[param.K3DIdentifier];

            if (entry && entry.model && view.K3DInstance.parameters.viewMode === viewModes.callback) {
                entry.model.send(_.extend({ msg_type: 'click_callback' }, param));
            }
        });

        resizeObserver.observe(el);
        el.addEventListener('contextmenu', contextMenuListener, true);

        // the HTML-snapshot button embeds the standalone source - fetched from the
        // kernel once per page, since no script URL points at it any more
        if (!window.k3dCompressed) {
            model.send({ msg_type: 'fetch_snapshot_source' });
        }

        view.evalInContext();
    });

    return () => {
        disposed = true;
        _.pull(REG.plots, view);

        resizeObserver.disconnect();
        el.removeEventListener('contextmenu', contextMenuListener, true);

        if (view.K3DInstance) {
            view.K3DInstance.off(view.K3DInstance.events.CAMERA_CHANGE, view.cameraChangeId);
            view.K3DInstance.off(view.K3DInstance.events.OBJECT_CHANGE, view.GUIObjectChanges);
            view.K3DInstance.off(view.K3DInstance.events.PARAMETERS_CHANGE, view.GUIParametersChanges);
            view.K3DInstance.off(view.K3DInstance.events.VOXELS_CALLBACK, view.voxelsCallback);
            view.K3DInstance.off(view.K3DInstance.events.OBJECT_HOVERED, view.objectHoverCallback);
            view.K3DInstance.off(view.K3DInstance.events.OBJECT_CLICKED, view.objectClickCallback);
        }

        if (view.cameraSyncTimeout !== null) {
            clearTimeout(view.cameraSyncTimeout);
            view.cameraSyncTimeout = null;
        }

        if (view.K3DInstance) {
            view.K3DInstance.disable();
        }
    };
}

/* ------------------------------------------------------------------------- */
/* transfer function editor                                                   */
/* ------------------------------------------------------------------------- */

function renderTFEditor({ model, el }) {
    const containerEnvelope = window.document.createElement('div');
    const container = window.document.createElement('div');

    containerEnvelope.style.cssText = [
        `height:${model.get('height')}px`,
        'position: relative',
    ].join(';');

    container.style.cssText = [
        'width: 100%',
        'height: 100%',
        'position: relative',
    ].join(';');

    containerEnvelope.appendChild(container);
    el.appendChild(containerEnvelope);

    let instance = null;
    // change events fire synchronously on set - the flag mutes the echo of our own edits
    let ownEdit = false;

    const resizeObserver = new ResizeObserver(() => {
        if (instance) {
            instance.refresh();
        }
    });

    whenConnected(el).then(() => {
        try {
            instance = new K3DTransferFunctionEditor(container, {
                height: model.get('height'),
                colorMap: Array.from(deserialized(model, 'color_map').data),
                opacityFunction: Array.from(deserialized(model, 'opacity_function').data),
            }, ((change) => {
                ownEdit = true;
                saveChanges(model, change.key, serialize.serialize({
                    data: new Float32Array(change.value),
                    shape: [change.value.length],
                }));
                ownEdit = false;
            }));
        } catch (e) {
            console.log(e);
            return;
        }

        model.on('change:color_map', () => {
            if (ownEdit || instance.isDragging()) {
                return;
            }
            instance.setColorMap(Array.from(deserialized(model, 'color_map').data));
        });

        model.on('change:opacity_function', () => {
            if (ownEdit || instance.isDragging()) {
                return;
            }
            instance.setOpacityFunction(Array.from(deserialized(model, 'opacity_function').data));
        });

        resizeObserver.observe(el);
    });

    return () => {
        resizeObserver.disconnect();
    };
}

/* ------------------------------------------------------------------------- */
/* dispatch                                                                   */
/* ------------------------------------------------------------------------- */

// objects and chunks ship a ~1KB stub _esm - the full module would otherwise ride in
// the synced state of every instance. The stub queues models until this module loads
// and adopts them; anything created later is adopted directly through REG.adopt.
REG.adopt = (model) => (
    model.get('_kind') === 'chunk' ? initChunk({ model }) : initObject({ model })
);

REG.pending.splice(0).forEach((entry) => {
    if (!entry.cancelled && !entry.adopted) {
        entry.adopted = true;
        entry.cleanup = REG.adopt(entry.model);
    }
});

export default {
    initialize(ctx) {
        const kind = ctx.model.get('_kind');

        if (kind === 'object') {
            return initObject(ctx);
        }

        if (kind === 'chunk') {
            return initChunk(ctx);
        }

        return undefined;
    },

    render(ctx) {
        const kind = ctx.model.get('_kind');

        if (kind === 'plot') {
            return renderPlot(ctx);
        }

        if (kind === 'tf_editor') {
            return renderTFEditor(ctx);
        }

        // objects and chunks are model-only widgets - nothing to show
        return undefined;
    },
};
