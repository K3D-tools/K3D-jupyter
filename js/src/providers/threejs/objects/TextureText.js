'use strict';

var closestPowOfTwo = require('./../helpers/Fn').closestPowOfTwo;

/**
 * Loader strategy to handle Text object
 * @method Text
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config) {
        config.visible = typeof(config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof(config.color) !== 'undefined' ? config.color : 0xFFFFFF;
        config.font_size = typeof(config.font_size) !== 'undefined' ? config.font_size : 68;
        config.font_weight = typeof(config.font_weight) !== 'undefined' ? config.font_weight : 700;

        var text = config.text.split('\n'),
            color = config.color,
            position = config.position,
            size = config.size || 1.0,

            // Setup font
            fontFace = config.font_face || 'Courier New',
            fontSize = config.font_size ,
            fontWeight = config.font_weight,
            fontSpec = fontWeight + ' ' + fontSize + 'px ' + fontFace,

            // Helper canvas
            canvas = document.createElement('canvas'),
            context = canvas.getContext('2d'),
            // Helpers
            isMultiline = text.length > 1,
            longestLineWidth,
            object;

        context.font = fontSpec;

        longestLineWidth = getLongestLineWidth(text, context);

        canvas.width = closestPowOfTwo(longestLineWidth);
        canvas.height = ~~canvas.width;

        context.font = fontSpec;
        context.textBaseline = 'top';
        context.fillStyle = colorToHex(color);
        context.lineWidth = 5;

        text.forEach(function (line, index) {
            var x = (canvas.width - longestLineWidth) / 2,
                y = canvas.height / 2 - (isMultiline ? fontSize : fontSize / 2) + (fontSize * index);

            context.strokeText(line, x, y);
            context.fillText(line, x, y);
        });

        object = getSprite(canvas, position, size);

        return Promise.resolve(object);
    }
};


/**
 * Gets the longest line
 * @inner
 * @memberof K3D.Providers.ThreeJS.Objects.Text
 * @param {Array} lines
 * @param {CanvasRenderingContext2D} context
 * @returns {*}
 */
function getLongestLineWidth(lines, context) {
    return lines.reduce(function (longest, text) {
        return Math.max(longest, context.measureText(text).width);
    }, 0);
}

/**
 * Get a THREE.Sprite based on helper canvas
 * @inner
 * @memberof K3D.Providers.ThreeJS.Objects.Text
 * @param {Element} canvas
 * @param {Array} position
 *
 * @returns {THREE.Sprite}
 */
function getSprite(canvas, position, size) {
    var texture = new THREE.Texture(canvas),
        material = new THREE.SpriteMaterial({map: texture}),
        sprite = new THREE.Sprite(material);

    texture.needsUpdate = true;

    sprite.position.set(position[0], position[1], position[2]);
    sprite.scale.set(size, size, size);
    sprite.boundingBox = new THREE.Box3().setFromCenterAndSize(
        new THREE.Vector3(),
        new THREE.Vector3(size, size, size)
    );
    sprite.updateMatrixWorld();

    return sprite;
}

function colorToHex(color) {
    color = parseInt(color, 10) + 0x1000000;

    return '#' + color.toString(16).substr(1);
}
