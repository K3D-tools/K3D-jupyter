const THREE = require('three');
const { closestPowOfTwo } = require('../helpers/Fn');
const { areAllChangesResolve } = require('../helpers/Fn');

/**
 * Loader strategy to handle Text object
 * @method Text
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.color = typeof (config.color) !== 'undefined' ? config.color : 0xFFFFFF;
        config.font_size = typeof (config.font_size) !== 'undefined' ? config.font_size : 68;
        config.font_weight = typeof (config.font_weight) !== 'undefined' ? config.font_weight : 700;

        const { color } = config;
        let { position } = config;
        const size = config.size || 1.0;

        // Setup font
        const fontFace = config.font_face || 'Courier New';
        const fontSize = config.font_size;
        const fontWeight = config.font_weight;
        const fontSpec = `${fontWeight} ${fontSize}px ${fontFace}`;
        const object = new THREE.Group();
        let i;

        if (position.data) {
            object.positions = position.data;
        } else {
            object.positions = position;
        }

        for (i = 0; i < object.positions.length / 3; i++) {
            // Helper canvas
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            context.font = fontSpec;

            let text = Array.isArray(config.text) ? config.text[i] : config.text;
            text = text.split('\n');
            const isMultiline = text.length > 1;

            const longestLineWidth = getLongestLineWidth(text, context);

            canvas.width = closestPowOfTwo(longestLineWidth);
            canvas.height = ~~canvas.width;

            context.font = fontSpec;
            context.textBaseline = 'top';
            context.fillStyle = colorToHex(color);
            context.lineWidth = 5;

            text.forEach((line, index) => {
                const x = (canvas.width - longestLineWidth) / 2;
                const y = canvas.height / 2 - (isMultiline ? fontSize : fontSize / 2) + (fontSize * index);

                context.strokeText(line, x, y);
                context.fillText(line, x, y);
            });

            const sprite = getSprite(canvas, object, i, size, config);
            object.add(sprite);
        }

        return Promise.resolve(object);
    },

    update(config, changes, obj) {
        const resolvedChanges = {};

        if (typeof (changes.position) !== 'undefined' && !changes.position.timeSeries) {
            let positions = changes.position.data || changes.position;

            obj.children.forEach(function (object, index) {
                object.position.fromArray(positions, index * 3);
            });

            resolvedChanges.position = null;
        }

        if (typeof (changes.size) !== 'undefined' && !changes.size.timeSeries) {
            obj.children.forEach(function (object) {
                object.scale.set(changes.size, changes.size, changes.size);
            });

            resolvedChanges.size = null;
        }

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({
                json: config,
                obj
            });
        }
        return false;
    },
};

function getLongestLineWidth(lines, context) {
    return lines.reduce((longest, text) => {
        const metric = context.measureText(text);
        let height = 0;

        if (metric.actualBoundingBoxAscent && metric.actualBoundingBoxDescent) {
            height = metric.actualBoundingBoxAscent + metric.actualBoundingBoxDescent;
        }

        return Math.max(longest, height, metric.width);
    }, 0);
}

function getSprite(canvas, object, index, size, config) {
    const texture = new THREE.Texture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(material);
    const modelMatrix = new THREE.Matrix4();

    texture.needsUpdate = true;

    sprite.position.fromArray(object.positions, index * 3);
    sprite.scale.set(size, size, size);
    sprite.boundingBox = new THREE.Box3().setFromCenterAndSize(
        new THREE.Vector3(),
        new THREE.Vector3(size, size, size),
    );
    if (config.model_matrix) {
        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        sprite.applyMatrix4(modelMatrix);
    }

    sprite.updateMatrixWorld();

    return sprite;
}

function colorToHex(color) {
    color = parseInt(color, 10) + 0x1000000;

    return `#${color.toString(16)
        .substr(1)}`;
}
