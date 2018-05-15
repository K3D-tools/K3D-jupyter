'use strict';

var map = {};
var counts = {};

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
        for (var i = 0; i < map[json.id].__controllers.length; i++) {
            var controller = map[json.id].__controllers[i];

            if (controller.property === param) {
                controller.object = json;
                controller.updateDisplay();

                return true;
            }
        }

        return false;
    }

    if (typeof(map[json.id]) === 'undefined') {
        counts[json.type] = counts[json.type] + 1 || 1;
        map[json.id] = objects.addFolder(json.type + ' #' + counts[json.type]);
    }

    var defaultParams = ['visible', 'outlines', 'wireframe', 'use_head', 'head_size', 'line_width', 'scale',
        'font_size', 'font_weight', 'size', 'point_size', 'level'];

    _.keys(json).forEach(function (param) {
            if (tryUpdate(json, param)) {
                return;
            }

            if (defaultParams.indexOf(param) !== -1) {
                map[json.id].add(json, param).onChange(change.bind(this, json, param));
            }

            // special dependencies
            if (param === 'color') {
                if (['Line', 'Points', 'VectorField', 'Vectors'].indexOf(json.type) === -1) {
                    map[json.id].addColor(json, param).onChange(change.bind(this, json, param));
                } else {
                    if (['Points', 'VectorField', 'Vectors'].indexOf(json.type) !== -1) {
                        if (typeof(json.colors) === 'undefined' || json.colors.length === 0) {
                            map[json.id].addColor(json, param).onChange(change.bind(this, json, param));
                        }
                    } else if (json.type === 'Line') {
                        if ((typeof(json.colors) === 'undefined' || json.colors.length === 0) &&
                            (typeof(json.attribute) === 'undefined' || json.attribute.length === 0) &&
                            (typeof(json.color_range) === 'undefined' || json.colors.color_range === 0) &&
                            (typeof(json.color_map) === 'undefined' || json.colors.color_map === 0)) {

                            map[json.id].addColor(json, param).onChange(change.bind(this, json, param));
                        }
                    }
                }
            }

            switch (param) {
                case 'origin_color':
                case 'head_color':
                    if (typeof(json.colors) === 'undefined') {
                        map[json.id].addColor(json, param).onChange(change.bind(this, json, param));
                    }
                    break;
                case 'outlines_color':
                    map[json.id].addColor(json, param).onChange(change.bind(this, json, param));
                    break;
                case 'text':
                    if (json.type !== 'STL') {
                        map[json.id].add(json, param).onChange(change.bind(this, json, param));
                    }
                    break;
                case 'shader':
                    if (json.type === 'Points') {
                        map[json.id].add(json, param, ['3dSpecular', '3d', 'flat', 'mesh']).onChange(
                            change.bind(this, json, param));
                    }
                    break;
                case 'width':
                    if (json.type === 'Line' && json.shader === 'mesh') {
                        map[json.id].add(json, param).onChange(change.bind(this, json, param));
                    }
                    break;
                case 'radial_segments':
                    if (json.shader === 'mesh') {
                        map[json.id].add(json, param, 0, 64, 1).onChange(change.bind(this, json, param));
                    }
                    break;
            }
        }
    );
}

module.exports = objectGUIProvider;
