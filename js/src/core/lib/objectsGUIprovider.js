//jshint maxstatements:false, maxcomplexity:false, maxdepth:false

'use strict';

function objectGUIProvider(K3D, json, objects, changes) {

    function addController(folder, obj, param, options1, options2, options3) {
        var controller = folder.add(obj, param, options1, options2, options3);
        folder.controllersMap[param] = controller;

        return controller;
    }

    function addColorController(folder, obj, param) {
        var controller = folder.addColor(obj, param);
        folder.controllersMap[param] = controller;

        return controller;
    }

    function change(json, key, value) {
        var change = {};

        change[key] = value;

        if (key !== 'name') {
            K3D.reload(json, change);
        }

        K3D.dispatch(K3D.events.OBJECT_CHANGE, {
            id: json.id,
            key: key,
            value: value
        });
    }

    function findControllers(json, param) {
        var folder = K3D.gui_map[json.id], main, low, high;

        if (param === 'color_range') {
            low = folder.controllersMap['_' + param + '_low'];
            high = folder.controllersMap['_' + param + '_high'];

            if (typeof(low) !== 'undefined') {
                return [low, high];
            }
        } else {
            main = folder.controllersMap[param];
            if (typeof(main) !== 'undefined') {
                return [main];
            }
        }

        return [];
    }

    function tryUpdate(json, param) {
        var controllers = findControllers(json, param);

        controllers.forEach(function (controller) {
            controller.object = json;
            controller.updateDisplay();
        });

        return controllers.length > 0;
    }

    if (typeof (K3D.gui_map) === 'undefined') {
        K3D.gui_map = {};
    }

    if (typeof (K3D.gui_counts) === 'undefined') {
        K3D.gui_counts = {};
    }

    if (typeof (K3D.gui_map[json.id]) === 'undefined') {
        K3D.gui_counts[json.type] = K3D.gui_counts[json.type] + 1 || 1;

        K3D.gui_map[json.id] = objects.addFolder(json.type + ' #' + K3D.gui_counts[json.type]);
        K3D.gui_map[json.id].controllersMap = {};
        K3D.gui_map[json.id].listenersId = K3D.on(K3D.events.OBJECT_REMOVED, function (id) {
            if (id === json.id) {
                var listenersId = K3D.gui_map[json.id].listenersId;
                var folder = K3D.gui_map[json.id];
                folder.close();
                objects.__ul.removeChild(folder.domElement.parentNode);
                delete objects.__folders[folder.name];
                objects.onResize();

                delete K3D.gui_map[json.id];

                K3D.off(K3D.events.OBJECT_REMOVED, listenersId);
            }
        });
    }

    var defaultParams = ['visible', 'outlines', 'wireframe', 'flat_shading', 'use_head', 'head_size', 'line_width',
        'scale', 'font_size', 'font_weight', 'size', 'point_size', 'level', 'samples', 'alpha_coef', 'gradient_step',
        'shadow_delay', 'focal_length', 'focal_plane'];

    var availableParams = defaultParams.concat(['color', 'origin_color', 'origin_color', 'head_color', 'outlines_color',
        'text', 'shader', 'shadow_res', 'shadow', 'ray_samples_count', 'width', 'radial_segments', 'mesh_detail',
        'opacity', 'color_range', 'name']);

    ((changes && Object.keys(changes)) || Object.keys(json)).forEach(function (param) {
            var colorMapLegendControllers, controller;

            if (availableParams.indexOf(param) === -1) {
                return;
            }

            if (param === 'name') {
                if (json.name === null) {
                    K3D.gui_map[json.id].name = json.name = json.type + ' #' + K3D.gui_counts[json.type];
                    change(json, 'name', json.name);
                } else {
                    K3D.gui_map[json.id].name = json.name;
                }
            }

            if (param === 'color_range' && json[param].length === 2) {
                json['_' + param + '_low'] = json[param][0];
                json['_' + param + '_high'] = json[param][1];

                // handle colorLegend
                if (K3D.parameters.colorbarObjectId === -1) { //auto
                    K3D.parameters.colorbarObjectId = json.id;
                }

                json.colorLegend = (K3D.parameters.colorbarObjectId === json.id);

                if (json.colorLegend) {
                    K3D.setColorMapLegend(json);
                }

                colorMapLegendControllers = findControllers(json, 'colorLegend');

                if (colorMapLegendControllers.length === 0) {
                    addController(K3D.gui_map[json.id], json, 'colorLegend').onChange(function (value) {
                        var objects = K3D.getWorld().ObjectsListJson;

                        Object.keys(objects).forEach(function (id) {
                            findControllers(objects[id], 'colorLegend').forEach(function (controller) {
                                objects[id].colorLegend = false;
                                controller.updateDisplay();
                            });
                        });

                        json.colorLegend = value;

                        if (value) {
                            K3D.setColorMapLegend(json);
                        } else {
                            K3D.setColorMapLegend(0);
                        }
                    });
                } else {
                    if (json.colorLegend) {
                        K3D.setColorMapLegend(json);
                    }
                }
            }

            if (tryUpdate(json, param)) {
                return;
            }

            if (defaultParams.indexOf(param) !== -1 && !json[param].timeSeries) {
                addController(K3D.gui_map[json.id], json, param).onChange(change.bind(this, json, param));
            }

            // special dependencies
            if (param === 'color') {
                if (['Line', 'Points', 'VectorField', 'Vectors'].indexOf(json.type) === -1) {
                    addColorController(K3D.gui_map[json.id], json, param).onChange(
                        change.bind(this, json, param));
                } else {
                    if (['Points', 'VectorField', 'Vectors'].indexOf(json.type) !== -1) {
                        if (typeof (json.colors) === 'undefined' || json.colors.length === 0) {
                            addColorController(K3D.gui_map[json.id], json, param).onChange(
                                change.bind(this, json, param));
                        }
                    } else if (json.type === 'Line') {
                        if ((typeof (json.colors) === 'undefined' || json.colors.length === 0) &&
                            (typeof (json.attribute) === 'undefined' || json.attribute.length === 0) &&
                            (typeof (json.color_range) === 'undefined' || json.colors.color_range === 0) &&
                            (typeof (json.color_map) === 'undefined' || json.colors.color_map === 0)) {

                            addColorController(K3D.gui_map[json.id], json, param).onChange(
                                change.bind(this, json, param));
                        }
                    }
                }
            }

            switch (param) {
                case 'origin_color':
                case 'head_color':
                    if (typeof (json.colors) === 'undefined') {
                        addColorController(K3D.gui_map[json.id], json, param).onChange(change.bind(this, json, param));
                    }
                    break;
                case 'outlines_color':
                    addColorController(K3D.gui_map[json.id], json, param).onChange(change.bind(this, json, param));
                    break;
                case 'text':
                    if (json.type !== 'STL' && !json[param].timeSeries) {
                        addController(K3D.gui_map[json.id], json, param).onChange(change.bind(this, json, param));
                    }
                    break;
                case 'shader':
                    if (json.type === 'Points') {
                        addController(K3D.gui_map[json.id], json, param,
                            ['3dSpecular', '3d', 'flat', 'mesh', 'dot']).onChange(
                            change.bind(this, json, param));
                    }
                    break;
                case 'shadow_res':
                    if (json.type === 'Volume') {
                        addController(K3D.gui_map[json.id], json, param, [32, 64, 128, 256, 512]).onChange(
                            change.bind(this, json, param));
                    }
                    break;
                case 'shadow':
                    if (json.type === 'Volume') {
                        addController(K3D.gui_map[json.id], json, param, ['off', 'on_demand', 'dynamic']).onChange(
                            change.bind(this, json, param));
                    }
                    break;
                case 'ray_samples_count':
                    if (json.type === 'Volume') {
                        addController(K3D.gui_map[json.id], json, param, [8, 16, 32, 64]).onChange(
                            change.bind(this, json, param));
                    }
                    break;
                case 'width':
                    if (json.type === 'Line' && json.shader === 'mesh') {
                        addController(K3D.gui_map[json.id], json, param).onChange(change.bind(this, json, param));
                    }
                    break;
                case 'radial_segments':
                    if (json.shader === 'mesh') {
                        addController(K3D.gui_map[json.id], json, param, 0, 64, 1).name('radialSeg').onChange(
                            change.bind(this, json, param));
                    }
                    break;
                case 'mesh_detail':
                    if (json.shader === 'mesh') {
                        addController(K3D.gui_map[json.id], json, param, 0, 8, 1).name('meshDetail').onChange(
                            change.bind(this, json, param));
                    }
                    break;
                case 'opacity':
                    if (!json[param].timeSeries) {
                        addController(K3D.gui_map[json.id], json, param, 0, 1.0).name('opacity').onChange(
                            change.bind(this, json, param));
                    }
                    break;
                case 'color_range':
                    if (json[param].length === 2) {
                        controller = addController(K3D.gui_map[json.id], json, '_' + param + '_low')
                            .name('vmin').onChange(function (value) {
                                json.color_range[0] = value;
                                change(json, 'color_range', json.color_range);
                            });

                        if (controller.__precision === 0 && controller.initialValue < 20) {
                            controller.__precision = 2;
                            controller.__impliedStep = 0.1;
                        }

                        controller = addController(K3D.gui_map[json.id], json, '_' + param + '_high')
                            .name('vmax').onChange(function (value) {
                                json.color_range[1] = value;
                                change(json, 'color_range', json.color_range);
                            });

                        if (controller.__precision === 0 && controller.initialValue < 20) {
                            controller.__precision = 2;
                            controller.__impliedStep = 0.1;
                        }
                    }
            }
        }
    );

    if (json.type === 'Volume') {
        if (findControllers(json, 'refreshLightMap').length === 0) {
            var obj = {
                refreshLightMap: function () {
                    K3D.getObjectById(json.id).refreshLightMap();
                    K3D.render();
                }
            };

            addController(K3D.gui_map[json.id], obj, 'refreshLightMap').name('Refresh light map');
        }
    }
}

module.exports = objectGUIProvider;
