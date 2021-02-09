'use strict';

var colorMapHelper = require('./core/lib/helpers/colorMap'),
    _ = require('./lodash'),
    semverRange = require('./version').version,
    serialize = require('./core/lib/helpers/serialize'),
    widgets = require('@jupyter-widgets/base');

function K3DTransferFunctionEditor(targetDOMNode, parameters, onChange) {
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'),
        opacityFunction = parameters.opacityFunction,
        svgNS = svg.namespaceURI,
        colorPicker,
        colorMap = parameters.colorMap,
        polygon,
        draggableElement,
        rect,
        topMargin = 10,
        bottomSection = 40,
        bottomSpacing = 10,
        selectedElement,
        opacityCircles = [],
        colormapCircles = [];

    require('style-loader?{attributes:{id: "k3d-style"}}!css-loader!./k3d.css');

    function removeOpacityCircle(evt) {
        var el = evt.target.parentNode,
            index = opacityCircles.indexOf(el);

        if (index === 0 || index === opacityCircles.length - 1) {
            return;
        }

        svg.removeChild(el);
        opacityCircles.splice(index, 1);
        opacityFunction.splice(index * 2, 2);
        refreshChart(svg, true);

        onChange({key: 'opacity_function', value: opacityFunction});
        evt.preventDefault();
    }

    function removeColormapCircle(evt) {
        var el = evt.target.parentNode,
            index = colormapCircles.indexOf(el);

        if (index === 0 || index === colormapCircles.length - 1) {
            return;
        }

        svg.removeChild(el);
        colormapCircles.splice(index, 1);
        colorMap.splice(index * 4, 4);
        refreshChart(svg, true);
        refreshRect(svg, true);

        onChange({key: 'color_map', value: colorMap});
        evt.preventDefault();
    }

    function makeCircle(cx, cy, r) {
        var circle = document.createElementNS(svg.namespaceURI, 'g');

        [{offset: 1, color: 'white'}, {offset: 0, color: 'black'}].forEach(function (conf) {
            var c = document.createElementNS(svg.namespaceURI, 'circle');

            c.setAttribute('cx', cx + conf.offset);
            c.setAttribute('cy', cy + conf.offset);
            c.setAttribute('r', r);
            c.setAttribute('stroke-width', 1);
            c.setAttribute('stroke', conf.color);
            c.setAttribute('fill', 'transparent');

            circle.appendChild(c);
        });

        return circle;
    }

    function makeDraggable(svg) {
        svg.addEventListener('mousedown', startDragOrAddNew);
        svg.addEventListener('mousemove', drag);
        svg.addEventListener('mouseup', endDrag);
        svg.addEventListener('mouseleave', endDrag);
        svg.addEventListener('touchstart', startDragOrAddNew);
        svg.addEventListener('touchmove', drag);
        svg.addEventListener('touchend', endDrag);
        svg.addEventListener('touchleave', endDrag);
        svg.addEventListener('touchcancel', endDrag);

        function getMousePosition(evt) {
            var CTM = svg.getScreenCTM();
            if (evt.touches) {
                evt = evt.touches[0];
            }
            return {
                x: (evt.clientX - CTM.e) / CTM.a,
                y: (evt.clientY - CTM.f) / CTM.d
            };
        }

        function startDragOrAddNew(evt) {
            if (evt.target.parentNode.classList.contains('draggable')) {
                selectedElement = draggableElement = evt.target.parentNode;
            } else {
                var coord = getMousePosition(evt);

                if (coord.y > topMargin && coord.y < svg.clientHeight - bottomSection) {
                    opacityFunction.push(colorMap[0] + coord.x / svg.clientWidth *
                        (colorMap[colorMap.length - 4] - colorMap[0]));
                    opacityFunction.push(1.0 - (coord.y - topMargin) / (svg.clientHeight - topMargin - bottomSection));

                    opacityFunction = ensureArraySorted(opacityFunction, 1);

                    refreshChart(svg, false);
                    onChange({key: 'opacity_function', value: opacityFunction});
                }

                if (coord.y > svg.clientHeight - bottomSection + bottomSpacing) {
                    var newX = colorMap[0] + coord.x / svg.clientWidth *
                        (colorMap[colorMap.length - 4] - colorMap[0]), i;
                    // new point injected
                    var data = colorMapHelper.mergeColorMapWithOpacity(colorMap,
                        [
                            colorMap[0], 0.0,
                            newX, 0.5,
                            colorMap[colorMap.length - 4], 1.0
                        ]
                    );

                    for (i = 0; i < data.length; i += 5) {
                        if (data[i] === newX) {
                            colorMap = colorMap.concat(data.slice(i, i + 4));
                        }
                    }

                    colorMap = ensureArraySorted(colorMap, 3);

                    refreshRect(svg, false);
                    onChange({key: 'color_map', value: colorMap});
                }
            }
        }

        function drag(evt) {
            var index;

            if (draggableElement) {
                evt.preventDefault();

                var coord = getMousePosition(evt);

                if (draggableElement.classList.contains('polygon')) {
                    index = opacityCircles.indexOf(draggableElement);

                    coord.y = Math.max(coord.y, topMargin);
                    coord.y = Math.min(coord.y, svg.clientHeight - bottomSection);

                    if (index === 0 || index === opacityCircles.length - 1) {
                        coord.x = parseFloat(draggableElement.childNodes[1].getAttribute('cx'));
                    } else {
                        coord.x = Math.min(coord.x,
                            parseFloat(opacityCircles[index + 1].childNodes[1].getAttribute('cx')) - 1
                        );
                        coord.x = Math.max(coord.x,
                            parseFloat(opacityCircles[index - 1].childNodes[1].getAttribute('cx')) + 1)
                        ;
                    }

                    opacityFunction[index * 2] = colorMap[0] + coord.x / svg.clientWidth *
                        (colorMap[colorMap.length - 4] - colorMap[0]);
                    opacityFunction[index * 2 + 1] = 1.0 - (coord.y - topMargin) /
                        (svg.clientHeight - topMargin - bottomSection);

                    onChange({key: 'opacity_function', value: opacityFunction});
                }

                if (draggableElement.classList.contains('colormap')) {
                    index = colormapCircles.indexOf(draggableElement);

                    coord.y = parseFloat(draggableElement.childNodes[1].getAttribute('cy'));

                    if (index === 0 || index === colormapCircles.length - 1) {
                        coord.x = parseFloat(draggableElement.childNodes[1].getAttribute('cx'));
                    } else {
                        coord.x = Math.min(coord.x,
                            parseFloat(colormapCircles[index + 1].childNodes[1].getAttribute('cx')) - 1
                        );
                        coord.x = Math.max(coord.x,
                            parseFloat(colormapCircles[index - 1].childNodes[1].getAttribute('cx')) + 1);
                    }

                    colorMap[index * 4] = colorMap[0] + coord.x / svg.clientWidth *
                        (colorMap[colorMap.length - 4] - colorMap[0]);

                    onChange({key: 'color_map', value: colorMap});
                }

                draggableElement.childNodes.forEach(function (el, index) {
                    el.setAttribute('cx', coord.x + 1 - index);
                    el.setAttribute('cy', coord.y + 1 - index);
                });

                refreshRect(svg, true);
                refreshChart(svg, true);
            }
        }

        function endDrag() {
            draggableElement = false;
        }
    }

    function ensureArraySorted(data, propertiesCount) {
        return data.reduce(function (prev, value, index) {
            if (index % (1 + propertiesCount) === 0) {
                prev.push({v: value, p: data.slice(index + 1, index + 1 + propertiesCount)});
            }

            return prev;
        }, []).sort(function (a, b) {
            return a.v - b.v;
        }).reduce(function (prev, value) {
            prev.push(value.v);
            value.p.forEach(function (v) {
                prev.push(v);
            });

            return prev;
        }, []);
    }

    function refreshRect(svg, skipCircles) {
        var i;

        colorMapHelper.createSVGGradient(svg, 'colorMap', colorMap, null, true);

        rect.setAttribute('x', 0);
        rect.setAttribute('y', svg.clientHeight - bottomSection + bottomSpacing);
        rect.setAttribute('width', svg.clientWidth);
        rect.setAttribute('height', bottomSection - bottomSpacing);

        if (!skipCircles) {
            colormapCircles.forEach(function (el) {
                svg.removeChild(el);
            });

            colormapCircles = [];

            for (i = 0; i < colorMap.length; i += 4) {
                var circle = makeCircle(
                    (colorMap[i] - colorMap[0]) / (colorMap[colorMap.length - 4] - colorMap[0]) * svg.clientWidth,
                    svg.clientHeight - (bottomSection - bottomSpacing) / 2,
                    5
                );
                circle.classList.add('draggable');
                circle.classList.add('colormap');

                circle.addEventListener('contextmenu', removeColormapCircle);
                circle.addEventListener('dblclick', function (evt) {
                    var index = colormapCircles.indexOf(evt.target.parentElement), r, g, b;

                    function getColor(evt) {
                        var hex = evt.target.value.substr(1);

                        colorMap[colorPicker.k3dIndex * 4 + 1] = parseInt(hex.substring(0, 2), 16) / 255;
                        colorMap[colorPicker.k3dIndex * 4 + 2] = parseInt(hex.substring(2, 4), 16) / 255;
                        colorMap[colorPicker.k3dIndex * 4 + 3] = parseInt(hex.substring(4, 6), 16) / 255;

                        refreshRect(svg, true);
                        refreshChart(svg, true);
                        onChange({key: 'color_map', value: colorMap});
                        colorPicker.removeEventListener('change', getColor);
                    }

                    r = '0' + Math.round(colorMap[index * 4 + 1] * 255).toString(16);
                    g = '0' + Math.round(colorMap[index * 4 + 2] * 255).toString(16);
                    b = '0' + Math.round(colorMap[index * 4 + 3] * 255).toString(16);

                    colorPicker.k3dIndex = index;
                    colorPicker.value = '#' + r.substr(-2) + g.substr(-2) + b.substr(-2);
                    colorPicker.click();

                    colorPicker.addEventListener('change', getColor);
                });

                svg.appendChild(circle);
                colormapCircles.push(circle);
            }
        }
    }

    function refreshChart(svg, skipCircles) {
        var path, i;

        path = opacityFunction.concat([colorMap[colorMap.length - 4], 0, colorMap[0], 0]);
        colorMapHelper.createSVGGradient(svg, 'transferFunction', colorMap, opacityFunction, true);

        polygon.points.clear();

        for (i = 0; i < path.length; i += 2) {
            var point = svg.createSVGPoint();
            point.x = (path[i] - colorMap[0]) / (colorMap[colorMap.length - 4] - colorMap[0]) * svg.clientWidth;
            point.y = (1.0 - path[i + 1]) * (svg.clientHeight - topMargin - bottomSection) + topMargin;
            polygon.points.appendItem(point);
        }

        polygon.setAttribute('fill', 'url(#transferFunction)');
        polygon.setAttribute('stroke-width', 1);
        polygon.setAttribute('stroke', 'black');

        if (!skipCircles) {
            opacityCircles.forEach(function (el) {
                svg.removeChild(el);
            });

            opacityCircles = [];

            for (i = 0; i < opacityFunction.length; i += 2) {
                var circle = makeCircle(
                    (opacityFunction[i] - colorMap[0]) / (colorMap[colorMap.length - 4] - colorMap[0]) * svg.clientWidth,
                    (1.0 - opacityFunction[i + 1]) *
                    (svg.clientHeight - topMargin - bottomSection) + topMargin,
                    5
                );

                circle.classList.add('draggable');
                circle.classList.add('polygon');

                circle.addEventListener('contextmenu', removeOpacityCircle);

                svg.appendChild(circle);
                opacityCircles.push(circle);
            }
        }
    }

    // set width and height
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.classList.add('k3d-transfer-function');

    polygon = document.createElementNS(svgNS, 'polygon');
    rect = document.createElementNS(svgNS, 'rect');
    rect.setAttribute('fill', 'url(#colorMap)');

    svg.appendChild(polygon);
    svg.appendChild(rect);

    colorPicker = document.createElement('input');
    colorPicker.setAttribute('type', 'color');
    colorPicker.classList.add('k3d-color-picker');

    targetDOMNode.appendChild(svg);
    targetDOMNode.appendChild(colorPicker);

    function refresh() {
        refreshChart(svg);
        refreshRect(svg);
    }

    window.addEventListener('resize', refresh, false);

    makeDraggable(svg);

    colorMap = ensureArraySorted(colorMap, 3);
    opacityFunction = ensureArraySorted(opacityFunction, 1);
    refresh();

    this.refresh = refresh;

    this.setColorMap = function (cm) {
        colorMap = cm;
        refresh();
    };

    this.setOpacityFunction = function (of) {
        opacityFunction = of;
        refresh();
    };

    this.getColorMap = function () {
        return colorMap;
    };

    this.getOpacityFunction = function () {
        return opacityFunction;
    };

    this.isDragging = function () {
        return draggableElement !== false;
    };
}

var transferFunctionModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(_.result({}, 'widgets.DOMWidgetModel.prototype.defaults'), {
        _model_name: 'TransferFunctionModel',
        _view_name: 'TransferFunctionView',
        _model_module: 'k3d',
        _view_module: 'k3d',
        _model_module_version: semverRange,
        _view_module_version: semverRange
    })
}, {
    serializers: _.extend({
        color_map: serialize,
        opacity_function: serialize
    }, widgets.DOMWidgetModel.serializers)
});

var transferFunctionView = widgets.DOMWidgetView.extend({
    render: function () {
        var containerEnvelope = window.document.createElement('div'),
            container = window.document.createElement('div');

        containerEnvelope.style.cssText = [
            'height:' + this.model.get('height') + 'px',
            'position: relative'
        ].join(';');

        container.style.cssText = [
            'width: 100%',
            'height: 100%',
            'position: relative'
        ].join(';');

        containerEnvelope.appendChild(container);
        this.el.appendChild(containerEnvelope);

        this.container = container;
        this.on('displayed', this._init, this);
    },

    remove: function () {

    },

    _init: function () {
        var self = this;

        this.model.on('change:color_map', this._setColorMap, this);
        this.model.on('change:opacity_function', this._setOpacityFunction, this);

        try {
            this.K3DTransferFunctionEditorInstance = new K3DTransferFunctionEditor(this.container, {
                height: this.model.get('height'),
                colorMap: Array.from(this.model.get('color_map').data),
                opacityFunction: Array.from(this.model.get('opacity_function').data)
            }, function (change) {
                self.model.set(change.key, {
                    data: new Float32Array(change.value),
                    shape: [change.value.length]
                }, {updated_view: self});
                self.model.save_changes();
            });
        } catch (e) {
            console.log(e);
            return;
        }
    },

    _setColorMap: function (widget, change, options) {
        var data;

        if (options.updated_view === this || this.K3DTransferFunctionEditorInstance.isDragging()) {
            return;
        }

        data = Array.from(this.model.get('color_map').data);
        this.K3DTransferFunctionEditorInstance.setColorMap(Array.from(data));
    },

    _setOpacityFunction: function (widget, change, options) {
        var data;

        if (options.updated_view === this || this.K3DTransferFunctionEditorInstance.isDragging()) {
            return;
        }

        data = this.model.get('opacity_function').data;
        this.K3DTransferFunctionEditorInstance.setOpacityFunction(Array.from(data));
    },

    processPhosphorMessage: function (msg) {
        widgets.DOMWidgetView.prototype.processPhosphorMessage.call(this, msg);
        switch (msg.type) {
            case 'after-attach':
                this.el.addEventListener('contextmenu', this, true);
                break;
            case 'before-detach':
                this.el.removeEventListener('contextmenu', this, true);
                break;
            case 'resize':
                this.handleResize(msg);
                break;
        }
    },

    handleEvent: function (event) {
        switch (event.type) {
            case 'contextmenu':
                this.handleContextMenu(event);
                break;
            default:
                widgets.DOMWidgetView.prototype.handleEvent.call(this, event);
                break;
        }
    },

    handleContextMenu: function () {
        // // Cancel context menu if on renderer:
        // if (this.container.contains(event.target)) {
        //     event.preventDefault();
        //     event.stopPropagation();
        // }
    },

    handleResize: function () {
        if (this.K3DTransferFunctionEditorInstance) {
            this.K3DTransferFunctionEditorInstance.refresh();
        }
    }
});

module.exports = {
    transferFunctionModel: transferFunctionModel,
    transferFunctionView: transferFunctionView,
    transferFunctionEditor: K3DTransferFunctionEditor
};
