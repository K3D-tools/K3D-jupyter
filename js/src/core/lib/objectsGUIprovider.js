'use strict';

function objectGUIProvider(K3D, json, objects) {

    function change(json, key, value) {
        K3D.reload(json);

        K3D.dispatch(K3D.events.OBJECT_CHANGE, {
            id: json.id,
            key: key,
            value: value
        });
    }

    function tryUpdate(json, param) {
        for (var i = 0; i < K3D.gui_map[json.id].__controllers.length; i++) {
            var controller = K3D.gui_map[json.id].__controllers[i];

            if (controller.property === param) {
                controller.object = json;
                controller.updateDisplay();

                return true;
            }
        }

        return false;
    }

    if (typeof(K3D.gui_map) === 'undefined') {
        K3D.gui_map = {};
    }

    if (typeof(K3D.gui_counts) === 'undefined') {
        K3D.gui_counts = {};
    }

    if (typeof(K3D.gui_map[json.id]) === 'undefined') {
        K3D.gui_counts[json.type] = K3D.gui_counts[json.type] + 1 || 1;
        K3D.gui_map[json.id] = objects.addFolder(json.type + ' #' + K3D.gui_counts[json.type]);
    }

    var defaultParams = ['visible', 'outlines', 'wireframe', 'use_head', 'head_size', 'line_width', 'scale',
        'font_size', 'font_weight', 'size', 'point_size', 'level'];

    _.keys(json).forEach(function (param) {
            if (tryUpdate(json, param)) {
                return;
            }

            if (defaultParams.indexOf(param) !== -1) {
                K3D.gui_map[json.id].add(json, param).onChange(change.bind(this, json, param));
            }

            // special dependencies
            if (param === 'color') {
                if (['Line', 'Points', 'VectorField', 'Vectors'].indexOf(json.type) === -1) {
                    K3D.gui_map[json.id].addColor(json, param).onChange(change.bind(this, json, param));
                } else {
                    if (['Points', 'VectorField', 'Vectors'].indexOf(json.type) !== -1) {
                        if (typeof(json.colors) === 'undefined' || json.colors.length === 0) {
                            K3D.gui_map[json.id].addColor(json, param).onChange(change.bind(this, json, param));
                        }
                    } else if (json.type === 'Line') {
                        if ((typeof(json.colors) === 'undefined' || json.colors.length === 0) &&
                            (typeof(json.attribute) === 'undefined' || json.attribute.length === 0) &&
                            (typeof(json.color_range) === 'undefined' || json.colors.color_range === 0) &&
                            (typeof(json.color_map) === 'undefined' || json.colors.color_map === 0)) {

                            K3D.gui_map[json.id].addColor(json, param).onChange(change.bind(this, json, param));
                        }
                    }
                }
            }

            switch (param) {
                case 'origin_color':
                case 'head_color':
                    if (typeof(json.colors) === 'undefined') {
                        K3D.gui_map[json.id].addColor(json, param).onChange(change.bind(this, json, param));
                    }
                    break;
                case 'outlines_color':
                    K3D.gui_map[json.id].addColor(json, param).onChange(change.bind(this, json, param));
                    break;
                case 'text':
                    if (json.type !== 'STL') {
                        K3D.gui_map[json.id].add(json, param).onChange(change.bind(this, json, param));
                    }
                    break;
                case 'shader':
                    if (json.type === 'Points') {
                        K3D.gui_map[json.id].add(json, param, ['3dSpecular', '3d', 'flat', 'mesh']).onChange(
                            change.bind(this, json, param));
                    }
                    break;
                case 'width':
                    if (json.type === 'Line' && json.shader === 'mesh') {
                        K3D.gui_map[json.id].add(json, param).onChange(change.bind(this, json, param));
                    }
                    break;
                case 'radial_segments':
                    if (json.shader === 'mesh') {
                        K3D.gui_map[json.id].add(json, param, 0, 64, 1).name('radialSeg').onChange(
                            change.bind(this, json, param));
                    }
                    break;
            }
        }
    );
}

module.exports = objectGUIProvider;
