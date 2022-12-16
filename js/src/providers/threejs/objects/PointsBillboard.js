const THREE = require('three');
const buffer = require('../../../core/lib/helpers/buffer');
const colorMapHelper = require('../../../core/lib/helpers/colorMap');
const Fn = require('../helpers/Fn');

const { commonUpdate } = Fn;
const { areAllChangesResolve } = Fn;
const { getColorsArray } = Fn;

/**
 * Loader strategy to handle Points object
 * @method Points
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config) {
        const modelMatrix = new THREE.Matrix4();
        const color = new THREE.Color(config.color);
        const pointPositions = config.positions.data;
        const pointColors = (config.colors && config.colors.data) || null;
        const { shader } = config;
        let colors = null;
        const opacities = (config.opacities && config.opacities.data
            && config.opacities.data.length === pointPositions.length / 3) ? config.opacities.data : null;
        const sizes = (config.point_sizes && config.point_sizes.data
            && config.point_sizes.data.length === pointPositions.length / 3) ? config.point_sizes.data : null;
        const { colorsToFloat32Array } = buffer;
        const fragmentShaderMap = {
            dot: require('./shaders/Points.dot.fragment.glsl'),
            flat: require('./shaders/Points.flat.fragment.glsl'),
            '3d': require('./shaders/Points.3d.fragment.glsl'),
            '3dspecular': require('./shaders/Points.3d.fragment.glsl'),
        };
        const vertexShaderMap = {
            dot: require('./shaders/Points.dot.vertex.glsl'),
            flat: require('./shaders/Points.vertex.glsl'),
            '3d': require('./shaders/Points.vertex.glsl'),
            '3dspecular': require('./shaders/Points.vertex.glsl'),
        };
        const colorMap = (config.color_map && config.color_map.data) || null;
        let opacityFunction = (config.opacity_function && config.opacity_function.data) || null;
        const colorRange = config.color_range;
        const attribute = (config.attribute && config.attribute.data) || null;
        let uniforms = {};
        let useColorMap = 0;

        const fragmentShader = fragmentShaderMap[shader.toLowerCase()] || fragmentShaderMap.flat;
        const vertexShader = vertexShaderMap[shader.toLowerCase()] || vertexShaderMap.flat;

        if (attribute && colorRange && colorMap && attribute.length > 0
            && colorRange.length > 0 && colorMap.length > 0) {
            useColorMap = 1;

            if (opacityFunction === null || opacityFunction.length === 0) {
                opacityFunction = [colorMap[0], 1.0, colorMap[colorMap.length - 4], 1.0];

                config.opacity_function = {
                    data: opacityFunction,
                    shape: [4],
                };
            }

            const canvas = colorMapHelper.createCanvasGradient(colorMap, 1024, opacityFunction);
            const colormap = new THREE.CanvasTexture(
                canvas,
                THREE.UVMapping,
                THREE.ClampToEdgeWrapping,
                THREE.ClampToEdgeWrapping,
                THREE.NearestFilter,
                THREE.NearestFilter,
            );
            colormap.needsUpdate = true;

            uniforms = {
                low: { value: colorRange[0] },
                high: { value: colorRange[1] },
                colormap: { type: 't', value: colormap },
            };
        } else {
            colors = (pointColors && pointColors.length === pointPositions.length / 3
                ? colorsToFloat32Array(pointColors) : getColorsArray(color, pointPositions.length / 3)
            );
        }

        const material = new THREE.ShaderMaterial({
            uniforms: THREE.UniformsUtils.merge([
                THREE.UniformsLib.lights,
                THREE.UniformsLib.points,
                uniforms,
            ]),
            defines: {
                USE_SPECULAR: (shader === '3dSpecular' ? 1 : 0),
                USE_COLOR_MAP: useColorMap,
                USE_PER_POINT_OPACITY: (opacities !== null ? 1 : 0),
                USE_PER_POINT_SIZE: (sizes !== null ? 1 : 0),
            },
            vertexShader,
            fragmentShader,
            opacity: config.opacity,
            depthWrite: (config.opacity === 1.0 && opacities === null),
            transparent: (config.opacity !== 1.0 || opacities !== null),
            lights: true,
            clipping: true,
            extensions: {
                fragDepth: true,
            },
        });

        // monkey-patching for imitate THREE.PointsMaterial
        material.size = config.point_size;
        material.color = new THREE.Color(1.0, 1.0, 1.0);
        material.map = null;
        material.isPointsMaterial = true;

        const object = new THREE.Points(
            getGeometry(pointPositions, colors, opacities, sizes, useColorMap ? attribute : null),
            material,
        );

        if (config.shader !== 'dot') {
            Fn.expandBoundingBox(object.geometry.boundingBox, config.point_size * 0.5);
        }

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix4(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },
    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        if (typeof (changes.positions) !== 'undefined' && !changes.positions.timeSeries
            && changes.positions.data.length === obj.geometry.attributes.position.array.length) {
            obj.geometry.attributes.position.array.set(changes.positions.data);
            obj.geometry.attributes.position.needsUpdate = true;

            obj.geometry.computeBoundingSphere();
            obj.geometry.computeBoundingBox();

            resolvedChanges.positions = null;
        }

        if (typeof (changes.color_range) !== 'undefined' && !changes.color_range.timeSeries
            && obj.geometry.attributes.attributes) {
            obj.material.uniforms.low.value = changes.color_range[0];
            obj.material.uniforms.high.value = changes.color_range[1];

            resolvedChanges.color_range = null;
        }

        if (((typeof (changes.color_map) !== 'undefined' && !changes.color_map.timeSeries)
                || (typeof (changes.opacity_function) !== 'undefined' && !changes.opacity_function.timeSeries))
            && obj.geometry.attributes.attributes) {
            const canvas = colorMapHelper.createCanvasGradient(
                (changes.color_map && changes.color_map.data) || config.color_map.data,
                1024,
                (changes.opacity_function && changes.opacity_function.data) || config.opacity_function.data,
            );

            obj.material.uniforms.colormap.value.image = canvas;
            obj.material.uniforms.colormap.value.needsUpdate = true;

            resolvedChanges.color_map = null;
            resolvedChanges.opacity_function = null;
        }

        if (typeof (changes.attribute) !== 'undefined' && !changes.attribute.timeSeries
            && obj.geometry.attributes.attributes
            && changes.attribute.data.length === obj.geometry.attributes.attributes.array.length) {
            obj.geometry.attributes.attributes.array.set(changes.attribute.data);
            obj.geometry.attributes.attributes.needsUpdate = true;

            resolvedChanges.attribute = null;
        }


        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};

/**
 * Setup BufferGeometry
 * @inner
 * @memberof K3D.Providers.ThreeJS.Objects.Points
 * @param  {Float32Array} positions
 * @param  {Float32Array} colors
 * @param  {Float32Array} opacities
 * @param  {Float32Array} sizes
 * @param  {Float32Array} attribute
 * @return {THREE.BufferGeometry}
 */
function getGeometry(positions, colors, opacities, sizes, attribute) {
    const geometry = new THREE.BufferGeometry();

    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));

    if (colors && colors.length > 0) {
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }

    if (opacities && opacities.length > 0) {
        geometry.setAttribute('opacities', new THREE.BufferAttribute(opacities, 1).setUsage(THREE.DynamicDrawUsage));
    }

    if (sizes && sizes.length > 0) {
        geometry.setAttribute('sizes', new THREE.BufferAttribute(sizes, 1).setUsage(THREE.DynamicDrawUsage));
    }

    if (attribute && attribute.length > 0) {
        geometry.setAttribute('attributes', new THREE.BufferAttribute(attribute, 1).setUsage(THREE.DynamicDrawUsage));
    }

    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    return geometry;
}
