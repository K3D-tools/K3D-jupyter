const mathHelper = require('./helpers/math');
const colorMapHelper = require('./helpers/colorMap');
const _ = require('../../lodash');

function getColorLegend(K3D, object) {
    let line;
    let text;
    let textShadow;
    let textGroup;
    let tick;
    const margin = 5;
    let majorScale;
    const range = [];
    const intervals = [];
    const texts = [];
    let maxTextWidth = 0;
    let intervalOffset;
    let intervalCount = 0;
    const strokeWidth = 0.5;
    let resizeListenerId = null;
    let i;

    if (!K3D.lastColorMap) {
        K3D.lastColorMap = {
            objectId: null,
            colorRange: [null, null],
            colorMap: null
        }
    }

    if (typeof (object) !== 'object') {
        return;
    }

    range[0] = Math.min(object.color_range[0], object.color_range[1]);
    range[1] = Math.max(object.color_range[0], object.color_range[1]);

    // avoid regenerating colormap
    if (K3D.lastColorMap.objectId === object.id &&
        K3D.lastColorMap.colorRange[0] == range[0] &&
        K3D.lastColorMap.colorRange[1] == range[1] &&
        _.isEqual(K3D.lastColorMap.colorMap, object.color_map.data)) {
        return;
    }


    if (K3D.colorMapNode) {
        K3D.getWorld().targetDOMNode.removeChild(K3D.colorMapNode);
        K3D.colorMapNode = null;
    }

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    const svgNS = svg.namespaceURI;
    const rect = document.createElementNS(svgNS, 'rect');

    svg.setAttribute('class', 'colorMapLegend');
    svg.setAttribute('viewBox', '0 0 100 100');
    svg.style.cssText = [
        'position: absolute',
        'bottom: 10px',
        'left: 10px',
        'width: 30vh',
        'height: 30vh',
        'max-width: 250px',
        'max-height: 250px',
        'min-width: 150px',
        'min-height: 150px',
        'z-index: 16777271',
        'pointer-events: none',
        'font-family: KaTeX_Main',
    ].join(';');

    colorMapHelper.createSVGGradient(svg, `colormap${object.id}`, object.color_map.data);

    rect.setAttribute('stroke-width', strokeWidth.toString(10));
    rect.setAttribute('stroke-linecap', 'square');
    rect.setAttribute('stroke', 'black');
    rect.setAttribute('fill', `url(#colormap${object.id})`);
    rect.setAttribute('width', (15 - margin).toString(10));
    rect.setAttribute('height', (100 - margin * 2).toString(10));
    rect.setAttribute('x', margin.toString(10));
    rect.setAttribute('y', margin.toString(10));

    svg.appendChild(rect);


    const colorRange = range[1] - range[0];
    majorScale = mathHelper.pow10ceil(Math.abs(colorRange)) / 10.0;

    while (intervalCount < 4) {
        intervalOffset = mathHelper.fmod(range[0], majorScale);
        intervalOffset = (intervalOffset > 0 ? majorScale - intervalOffset : 0);
        intervalCount = Math.floor((Math.abs(colorRange) - intervalOffset + majorScale * 0.001) / majorScale);

        if (intervalCount < 4) {
            majorScale /= 2.0;
        }
    }

    for (i = 0; i <= intervalCount; i++) {
        intervals.push(range[0] + intervalOffset + i * majorScale);
    }

    intervals.forEach((v) => {
        textGroup = document.createElementNS(svgNS, 'g');
        line = document.createElementNS(svgNS, 'line');
        text = document.createElementNS(svgNS, 'text');
        textShadow = document.createElementNS(svgNS, 'text');

        const y = margin + (100 - margin * 2) * (1.0 - (v - range[0]) / colorRange);

        if (K3D.parameters.colorbarScientific) {
            tick = v.toPrecision(4);
        } else {
            tick = v.toFixed((majorScale.toString(10).split('.')[1] || '').length);
        }

        line.setAttribute('x1', '13');
        line.setAttribute('y1', y.toString(10));
        line.setAttribute('x2', '17');
        line.setAttribute('y2', y.toString(10));
        line.setAttribute('stroke-width', strokeWidth.toString(10));
        line.setAttribute('stroke', 'black');
        svg.appendChild(line);

        text.setAttribute('x', '0');
        text.setAttribute('y', '0');
        text.setAttribute('alignment-baseline', 'middle');
        text.setAttribute('text-anchor', 'end');
        text.setAttribute('font-size', '0.5em');
        text.setAttribute('fill', 'rgb(68, 68, 68)');
        text.innerHTML = tick;

        textShadow.setAttribute('x', '0.5');
        textShadow.setAttribute('y', '0.5');
        textShadow.setAttribute('alignment-baseline', 'middle');
        textShadow.setAttribute('text-anchor', 'end');
        textShadow.setAttribute('font-size', '0.5em');
        textShadow.setAttribute('fill', 'rgb(255, 255, 255)');
        textShadow.innerHTML = tick;

        textGroup.setAttribute('pos_y', y.toString(10));

        textGroup.appendChild(textShadow);
        textGroup.appendChild(text);

        texts.push(textGroup);
        svg.appendChild(textGroup);
    });

    K3D.getWorld().targetDOMNode.appendChild(svg);

    function tryPosLabels() {
        if (K3D.getWorld().width < 10 || K3D.getWorld().height < 10) {
            if (resizeListenerId === null) {
                resizeListenerId = K3D.on(K3D.events.RESIZED, () => {
                    tryPosLabels();
                });
            }
        } else {
            if (resizeListenerId !== null) {
                K3D.off(K3D.events.RESIZED, resizeListenerId);
                resizeListenerId = null;
            }

            maxTextWidth = texts.reduce((max, t) => Math.max(max, t.getBBox().width), 0);

            texts.forEach((t) => {
                const x = (maxTextWidth + 20).toString(10);
                const y = t.getAttribute('pos_y');

                t.setAttribute('transform', `translate(${x}, ${y})`);
            });
        }
    }

    tryPosLabels();

    K3D.colorMapNode = svg;

    K3D.lastColorMap = {
        objectId: object.id,
        colorRange: range,
        colorMap: object.color_map.data
    }
}

module.exports = {
    getColorLegend,
};
