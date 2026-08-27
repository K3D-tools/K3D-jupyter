const THREE = require('three');
const { colorsToFloat32Array } = require('../../../core/lib/helpers/buffer');
const colorMapHelper = require('../../../core/lib/helpers/colorMap');
const Fn = require('../helpers/Fn');

const { commonUpdate } = Fn;
const { areAllChangesResolve } = Fn;
const { computeFiniteBounds } = Fn;
const { getColorsArray } = Fn;
const { handleColorMap } = Fn;

/**
 * Loader strategy to handle Line object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config) {
        const geometry = new THREE.BufferGeometry();
        const material = new THREE.MeshBasicMaterial({
            opacity: config.opacity,
            depthWrite: config.opacity === 1.0,
            transparent: config.opacity !== 1.0,
        });
        const verticesColors = (config.colors && config.colors.data) || null;
        const color = new THREE.Color(config.color);
        let colors;
        const colorRange = config.color_range;
        const colorMap = (config.color_map && config.color_map.data) || null;
        const attribute = (config.attribute && config.attribute.data) || null;
        const object = new THREE.Line(geometry, material);
        const modelMatrix = new THREE.Matrix4();
        const position = config.vertices.data;

        if (attribute && colorRange && colorMap && attribute.length > 0 && colorRange.length > 0
            && colorMap.length > 0) {
            handleColorMap(geometry, colorMap, colorRange, attribute, material);
        } else {
            colors = (verticesColors && verticesColors.length === position.length / 3
                    ? colorsToFloat32Array(verticesColors) : getColorsArray(color, position.length / 3)
            );

            material.setValues({ vertexColors: true });
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));

        computeFiniteBounds(geometry);

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix4(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        if (typeof (obj.geometry.attributes.uv) !== 'undefined') {
            if (typeof (changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
                const attribute = (config.attribute && config.attribute.data) || null;
                const uv = obj.geometry.attributes.uv.array;

                if (attribute && attribute.length === uv.length) {
                    const low = changes.color_range[0];
                    const span = changes.color_range[1] - low;

                    for (let i = 0; i < uv.length; i++) {
                        uv[i] = (attribute[i] - low) / span;
                    }

                    obj.geometry.attributes.uv.needsUpdate = true;
                    resolvedChanges.color_range = null;
                }
            }

            if (typeof (changes.attribute) !== 'undefined' && !changes.attribute.timeSeries
                && changes.attribute.data.length === obj.geometry.attributes.uv.array.length) {
                const data = obj.geometry.attributes.uv.array;

                for (let i = 0; i < data.length; i++) {
                    data[i] = (changes.attribute.data[i] - config.color_range[0])
                        / (config.color_range[1] - config.color_range[0]);
                }

                obj.geometry.attributes.uv.needsUpdate = true;
                resolvedChanges.attribute = null;
            }
        }

        if (((typeof (changes.colors) !== 'undefined' && !changes.colors.timeSeries)
            || (typeof (changes.color) !== 'undefined' && !changes.color.timeSeries))
            && obj.geometry.attributes.color) {
            const count = obj.geometry.attributes.color.array.length / 3;
            const verticesColors = (changes.colors && changes.colors.data)
                || (config.colors && config.colors.data) || null;

            obj.geometry.attributes.color.array.set(
                verticesColors && verticesColors.length === count
                    ? colorsToFloat32Array(verticesColors)
                    : getColorsArray(new THREE.Color(config.color), count),
            );
            obj.geometry.attributes.color.needsUpdate = true;

            resolvedChanges.colors = null;
            resolvedChanges.color = null;
        }

        if (typeof (changes.color_map) !== 'undefined' && !changes.color_map.timeSeries
            && obj.material.map) {
            obj.material.map.image = colorMapHelper.createCanvasGradient(changes.color_map.data, 1024, 1);
            obj.material.map.needsUpdate = true;

            resolvedChanges.color_map = null;
        }

        if (typeof (changes.vertices) !== 'undefined' && !changes.vertices.timeSeries
            && changes.vertices.data.length === obj.geometry.attributes.position.array.length) {
            obj.geometry.attributes.position.array.set(changes.vertices.data);
            obj.geometry.attributes.position.needsUpdate = true;

            computeFiniteBounds(obj.geometry);

            resolvedChanges.vertices = null;
        }

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};
