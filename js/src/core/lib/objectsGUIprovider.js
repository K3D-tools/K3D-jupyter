// jshint maxstatements:false, maxcomplexity:false, maxdepth:false

const planeHelper = require('./helpers/planeGUI');

function changeParameter(K3D, json, key, value, timeSeriesReload) {
    const change = {};

    change[key] = value;

    if (key !== 'name') {
        K3D.reload(json, change, timeSeriesReload);
    }

    K3D.dispatch(K3D.events.OBJECT_CHANGE, {
        id: json.id,
        key,
        value,
    });
}

function update(K3D, json, GUI, changes) {
    let sliceViewerControllers;

    function moveToGroup(config) {
        let parent;

        if (config.group != null) {
            if (typeof (K3D.gui_groups[config.group]) === 'undefined') {
                K3D.gui_groups[config.group] = GUI.addFolder(`${config.group}`).close();
            }

            parent = K3D.gui_groups[config.group];
        } else {
            parent = GUI;
        }

        // cleanup previous plance
        K3D.gui_map[config.id].parent.children.splice(K3D.gui_map[config.id].parent.children.indexOf(
            K3D.gui_map[config.id],
        ), 1);
        K3D.gui_map[config.id].parent.folders.splice(K3D.gui_map[config.id].parent.folders.indexOf(
            K3D.gui_map[config.id],
        ), 1);

        K3D.gui_map[config.id].parent = parent;
        K3D.gui_map[config.id].parent.$children.append(K3D.gui_map[config.id].domElement);

        // add to new place
        if (parent.children.indexOf(K3D.gui_map[config.id]) === -1) {
            parent.children.push(K3D.gui_map[config.id]);
        }

        if (parent.folders.indexOf(K3D.gui_map[config.id]) === -1) {
            parent.folders.push(K3D.gui_map[config.id]);
        }

        // remove empty groups
        Object.keys(K3D.gui_groups).forEach((group) => {
            if (K3D.gui_groups[group].children.length === 0) {
                K3D.gui_groups[group].destroy();
                delete K3D.gui_groups[group];
            }
        });
    }

    function addController(folder, obj, param, options1, options2, options3) {
        const controller = folder.add(obj, param, options1, options2, options3);
        folder.controllersMap[param] = controller;

        return controller;
    }

    function addColorController(folder, obj, param) {
        const controller = folder.addColor(obj, param);
        folder.controllersMap[param] = controller;

        return controller;
    }

    function findControllers(param) {
        const folder = K3D.gui_map[json.id];
        let main;
        let low;
        let high;

        if (param === 'color_range') {
            low = folder.controllersMap[`_${param}_low`];
            high = folder.controllersMap[`_${param}_high`];

            if (typeof (low) !== 'undefined') {
                return [low, high];
            }
        } else {
            main = folder.controllersMap[param];
            if (typeof (main) !== 'undefined') {
                return [main];
            }
        }

        return [];
    }

    function tryUpdate(param) {
        const controllers = findControllers(param);

        controllers.forEach((controller) => {
            controller.object = json;
            controller.updateDisplay();
        });

        return controllers.length > 0;
    }

    if (!GUI) {
        return;
    }

    if (typeof (K3D.gui_map) === 'undefined') {
        K3D.gui_map = {};
    }

    if (typeof (K3D.gui_groups) === 'undefined') {
        K3D.gui_groups = {};
    }

    if (typeof (K3D.gui_counts) === 'undefined') {
        K3D.gui_counts = {};
    }

    if (typeof (K3D.gui_map[json.id]) === 'undefined') {
        K3D.gui_counts[json.type] = K3D.gui_counts[json.type] + 1 || 1;
        let parent;

        if (json.group === null) {
            parent = GUI;
        } else {
            if (typeof (K3D.gui_groups[json.group]) === 'undefined') {
                K3D.gui_groups[json.group] = GUI.addFolder(`${json.group}`).close();
            }
            parent = K3D.gui_groups[json.group];
        }

        K3D.gui_map[json.id] = parent.addFolder(`${json.type} #${K3D.gui_counts[json.type]}`).close();

        K3D.gui_map[json.id].controllersMap = {};
        K3D.gui_map[json.id].listenersId = K3D.on(K3D.events.OBJECT_REMOVED, (id) => {
            if (id === json.id.toString()) {
                const {listenersId} = K3D.gui_map[json.id];
                const folder = K3D.gui_map[json.id];

                folder.destroy();

                if (json.group !== null && K3D.gui_groups[json.group]) {
                    if (K3D.gui_groups[json.group].children.length === 0) {
                        K3D.gui_groups[json.group].destroy();
                        delete K3D.gui_groups[json.group];
                    }
                }

                delete K3D.gui_map[json.id];

                K3D.off(K3D.events.OBJECT_REMOVED, listenersId);
            }
        });
    }

    const defaultParams = ['visible', 'outlines', 'wireframe', 'flat_shading', 'use_head', 'head_size', 'line_width',
        'scale', 'font_size', 'font_weight', 'size', 'point_size', 'level', 'samples', 'alpha_coef', 'gradient_step',
        'shadow_delay', 'focal_length', 'focal_plane', 'on_top', 'max_length', 'label_box', 'is_html', 'shininess, mask_opacity'];

    const availableParams = defaultParams.concat(['color', 'origin_color', 'origin_color', 'head_color',
        'outlines_color', 'text', 'shader', 'shadow_res', 'shadow', 'ray_samples_count', 'width', 'radial_segments',
        'mesh_detail', 'opacity', 'color_range', 'name', 'group', 'color_map', 'mode',
        'direction', 'slice_x', 'slice_y', 'slice_z', 'volumeSliceMask']);

    // handle sliceViewer
    if (json.type === 'VolumeSlice') {
        if (K3D.parameters.sliceViewerObjectId === -1) { // auto
            K3D.setSliceViewer(json);
            json.sliceViewer = true;
        }

        sliceViewerControllers = findControllers('sliceViewer');

        if (sliceViewerControllers.length === 0) {
            json.sliceViewer = (K3D.parameters.sliceViewerObjectId === json.id);
            addController(K3D.gui_map[json.id], json, 'sliceViewer').onChange((value) => {
                json.sliceViewer = value;

                if (value) {
                    K3D.setSliceViewer(json);
                } else {
                    K3D.setSliceViewer(0);
                }
            });
        }

        json.sliceViewer = (K3D.parameters.sliceViewerObjectId === json.id);
    }

    ((changes && Object.keys(changes)) || Object.keys(json)).forEach(function (param) {
        let colorMapLegendControllers;
        let controller;

        if (availableParams.indexOf(param) === -1) {
            return;
        }

        if (param === 'name') {
            if (json.name === null) {
                json.name = `${json.type} #${K3D.gui_counts[json.type]}`;
                K3D.gui_map[json.id].title(json.name);

                changeParameter(K3D, json, 'name', json.name);
            } else {
                K3D.gui_map[json.id].title(json.name);
            }
        }

        if (param === 'group') {
            moveToGroup(json);
        }

        if (param === 'color_range' && json[param].length >= 2) {
            json[`_${param}_low`] = json[param][0];
            json[`_${param}_high`] = json[param][1];

            // handle colorLegend
            colorMapLegendControllers = findControllers('colorLegend');

            if (colorMapLegendControllers.length === 0) {
                json.colorLegend = (K3D.parameters.colorbarObjectId === json.id);
                addController(K3D.gui_map[json.id], json, 'colorLegend').onChange((value) => {
                    json.colorLegend = value;

                    if (value) {
                        K3D.setColorMapLegend(json);
                    } else {
                        K3D.setColorMapLegend(0);
                    }
                });
            }
        }

        json.colorLegend = (K3D.parameters.colorbarObjectId === json.id);

        if (param === 'color_map' || param === 'color_range') {
            if (json.colorLegend) {
                K3D.setColorMapLegend(json);
            }
        }

        if (tryUpdate(param)) {
            return;
        }

        if (defaultParams.indexOf(param) !== -1 && !json[param].timeSeries) {
            addController(K3D.gui_map[json.id], json, param).onChange(changeParameter.bind(this, K3D, json, param));
        }

        // special dependencies
        if (param === 'color') {
            if (['Line', 'Points', 'VectorField', 'Vectors'].indexOf(json.type) === -1) {
                addColorController(K3D.gui_map[json.id], json, param).onChange(
                    changeParameter.bind(this, K3D, json, param),
                );
            } else if (['Points', 'VectorField', 'Vectors'].indexOf(json.type) !== -1) {
                if (typeof (json.colors) === 'undefined' || json.colors.length === 0) {
                    addColorController(K3D.gui_map[json.id], json, param).onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
            } else if (json.type === 'Line') {
                if ((typeof (json.colors) === 'undefined' || json.colors.length === 0)
                    && (typeof (json.attribute) === 'undefined' || json.attribute.length === 0)
                    && (typeof (json.color_range) === 'undefined' || json.colors.color_range === 0)
                    && (typeof (json.color_map) === 'undefined' || json.colors.color_map === 0)) {
                    addColorController(K3D.gui_map[json.id], json, param).onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
            }
        }

        if (json[param] !== null && json[param].timeSeries) {
            return;
        }

        switch (param) {
            case 'origin_color':
            case 'head_color':
                if (typeof (json.colors) === 'undefined') {
                    addColorController(K3D.gui_map[json.id], json, param).onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
                break;
            case 'outlines_color':
                addColorController(K3D.gui_map[json.id], json, param).onChange(
                    changeParameter.bind(this, K3D, json, param),
                );
                break;
            case 'text':
                if (json.type !== 'STL' && !Array.isArray(json.text)) {
                    addController(K3D.gui_map[json.id], json, param).onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
                break;
            case 'shader':
                if (json.type === 'Points') {
                    addController(
                        K3D.gui_map[json.id],
                        json,
                        param,
                        ['3dSpecular', '3d', 'flat', 'mesh', 'dot'],
                    ).onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }

                if (json.type === 'Lines' || json.type === 'Line') {
                    addController(
                        K3D.gui_map[json.id],
                        json,
                        param,
                        ['simple', 'thick', 'mesh'],
                    ).onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
                break;
            case 'mode':
                if (json.type === 'Label') {
                    addController(
                        K3D.gui_map[json.id],
                        json,
                        param,
                        ['dynamic', 'local', 'side'],
                    ).onChange(changeParameter.bind(this, K3D, json, param));
                }
                break;
            case 'slice_x':
                if (json.type === 'VolumeSlice') {
                    let shape = Array.isArray(json.volume) ? json.volume[0].shape : json.volume.shape;

                    addController(K3D.gui_map[json.id], json, param, -1, shape[2] - 1, 1).onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
                break;
            case 'slice_y':
                if (json.type === 'VolumeSlice') {
                    let shape = Array.isArray(json.volume) ? json.volume[0].shape : json.volume.shape;

                    addController(K3D.gui_map[json.id], json, param, -1, shape[1] - 1, 1).onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
                break;
            case 'slice_z':
                if (json.type === 'VolumeSlice') {
                    let shape = Array.isArray(json.volume) ? json.volume[0].shape : json.volume.shape;

                    addController(K3D.gui_map[json.id], json, param, -1, shape[0] - 1, 1).onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
                break;
            case 'shadow_res':
                if (json.type === 'Volume') {
                    addController(K3D.gui_map[json.id], json, param, [32, 64, 128, 256, 512]).onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
                break;
            case 'shadow':
                if (json.type === 'Volume') {
                    addController(K3D.gui_map[json.id], json, param, ['off', 'on_demand', 'dynamic']).onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
                break;
            case 'ray_samples_count':
                if (json.type === 'Volume') {
                    addController(K3D.gui_map[json.id], json, param, [8, 16, 32, 64]).onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
                break;
            case 'width':
                if ((json.type === 'Line' || json.type === 'Lines') && json.shader === 'mesh') {
                    addController(K3D.gui_map[json.id], json, param).onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
                break;
            case 'radial_segments':
                addController(K3D.gui_map[json.id], json, param, 0, 64, 1).name('radialSeg').onChange(
                    changeParameter.bind(this, K3D, json, param),
                );
                break;
            case 'mesh_detail':
                if (json.shader === 'mesh') {
                    addController(K3D.gui_map[json.id], json, param, 0, 12, 1).name('meshDetail').onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
                break;
            case 'opacity':
                if (!json[param].timeSeries) {
                    addController(K3D.gui_map[json.id], json, param, 0, 1.0).name('opacity').onChange(
                        changeParameter.bind(this, K3D, json, param),
                    );
                }
                break;
            case 'color_range':
                if (json[param].length === 2) {
                    controller = addController(K3D.gui_map[json.id], json, `_${param}_low`)
                        .name('vmin').onChange((value) => {
                            json.color_range[0] = value;
                            changeParameter(K3D, json, 'color_range', json.color_range);
                        });

                    if (controller.initialValue < 20) {
                        controller._step = 0.1;
                    }

                    controller = addController(K3D.gui_map[json.id], json, `_${param}_high`)
                        .name('vmax').onChange((value) => {
                            json.color_range[1] = value;
                            changeParameter(K3D, json, 'color_range', json.color_range);
                        });

                    if (controller.initialValue < 20) {
                        controller._step = 0.1;
                    }
                }
                break;
            default:
                break;
        }
    });

    if (json.type === 'Mesh') {
        if (typeof (K3D.gui_map[json.id].controllersMap.volumeSliceMask) === 'undefined') {
            json.volumeSliceMask = K3D.parameters.sliceViewerMaskObjectIds.indexOf(json.id) !== -1;

            addController(K3D.gui_map[json.id], json, 'volumeSliceMask').name('Slice mask').onChange((value) => {
                let ids = K3D.parameters.sliceViewerMaskObjectIds.slice();

                json.volumeSliceMask = value;

                if (value) {
                    if (ids.indexOf(json.id) === -1) {
                        ids.push(json.id);
                    }
                } else {
                    ids = ids.filter((id) => id !== json.id);
                }

                K3D.setSliceViewerMaskObjects(ids);
            });
        }

        if (typeof (K3D.gui_map[json.id].controllersMap.slice_planes) === 'undefined') {
            K3D.gui_map[json.id].controllersMap.slice_planes = K3D.gui_map[json.id].addFolder('Slice planes');
        }

        planeHelper.init(
            K3D,
            K3D.gui_map[json.id].controllersMap.slice_planes,
            json.slice_planes,
            'slicePlanes',
            changeParameter.bind(null, K3D, json, 'slice_planes'),
        );
    }

    if (json.type === 'Volume') {
        if (findControllers('refreshLightMap').length === 0) {
            const obj = {
                refreshLightMap() {
                    K3D.getObjectById(json.id).refreshLightMap();
                    K3D.render();
                },
            };

            addController(K3D.gui_map[json.id], obj, 'refreshLightMap').name('Refresh light map');
        }
    }
}

module.exports = {
    update,
    changeParameter,
};
