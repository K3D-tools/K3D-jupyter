const THREE = require('three');
const buffer = require('../../../core/lib/helpers/buffer');
const colorMapHelper = require('../../../core/lib/helpers/colorMap');
const interactionsHelper = require('../helpers/Interactions');
const pointsCallback = require('../interactions/PointsCallback');
const pointsIntersect = require('../interactions/PointsIntersect');
const Fn = require('../helpers/Fn');

const { areAllChangesResolve } = Fn;
const { commonUpdate } = Fn;
const { getColorsArray } = Fn;

/**
 * Loader strategy to handle Points object
 * @method PointsMesh
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configuration params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config, K3D) {
        config.shininess = typeof (config.shininess) !== 'undefined' ? config.shininess : 50.0;

        const modelMatrix = new THREE.Matrix4();
        const color = new THREE.Color(config.color);
        const positions = config.positions.data;
        const pointColors = (config.colors && config.colors.data) || null;
        const meshDetail = typeof (config.mesh_detail) !== 'undefined' ? config.mesh_detail : 2;
        let colors;
        const opacities = (config.opacities && config.opacities.data
            && config.opacities.data.length === positions.length / 3) ? config.opacities.data : null;
        const sizes = (config.point_sizes && config.point_sizes.data
            && config.point_sizes.data.length === positions.length / 3) ? config.point_sizes.data : null;
        const { colorsToFloat32Array } = buffer;
        const phongShader = THREE.ShaderLib.phong;
        let i;
        const boundingBoxGeometry = new THREE.BufferGeometry();
        const geometry = new THREE.IcosahedronBufferGeometry(config.point_size * 0.5, meshDetail);
        const colorMap = (config.color_map && config.color_map.data) || null;
        let opacityFunction = (config.opacity_function && config.opacity_function.data) || null;
        const colorRange = config.color_range;
        const attribute = (config.attribute && config.attribute.data) || null;
        let uniforms = {};
        let useColorMap = 0;

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

            const canvas = colorMapHelper.createCanvasGradient(colorMap, 1024, 1, opacityFunction);
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
            geometry.setAttribute(
                'attributes',
                new THREE.InstancedBufferAttribute(attribute, 1).setUsage(THREE.DynamicDrawUsage),
            );
        } else {
            colors = (pointColors && pointColors.length === positions.length / 3
                    ? colorsToFloat32Array(pointColors) : getColorsArray(color, positions.length / 3)
            );
        }

        geometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array(colors), 3));

        if (opacities) {
            geometry.setAttribute(
                'opacities',
                new THREE.InstancedBufferAttribute(opacities, 1),
            );
        }

        // boundingBox & boundingSphere
        boundingBoxGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        boundingBoxGeometry.computeBoundingSphere();
        boundingBoxGeometry.computeBoundingBox();
        Fn.expandBoundingBox(boundingBoxGeometry.boundingBox, config.point_size * 0.5);

        geometry.boundingBox = boundingBoxGeometry.boundingBox.clone();

        const material = new THREE.ShaderMaterial({
            uniforms: THREE.UniformsUtils.merge([phongShader.uniforms, {
                shininess: { value: config.shininess },
                opacity: { value: config.opacity },
            }, uniforms]),
            defines: {
                USE_PER_POINT_OPACITY: (opacities !== null ? 1 : 0),
                USE_COLOR_MAP: useColorMap
            },
            vertexShader: require('./shaders/PointsMesh.vertex.glsl'),
            fragmentShader: require('./shaders/PointsMesh.fragment.glsl'),
            transparent: (config.opacity !== 1.0 || opacities !== null),
            lights: true,
            clipping: true,
            vertexColors: THREE.VertexColors,
        });

        if (K3D.parameters.depthPeels === 0) {
            material.depthWrite = (config.opacity === 1.0 && opacities === null);
        } else {
            material.onBeforeCompile = K3D.colorOnBeforeCompile;
        }

        const object = new THREE.InstancedMesh(geometry, material, positions.length / 3);
        object.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

        let pointsGeometry = new THREE.BufferGeometry();
        pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix4(modelMatrix);
        object.updateMatrixWorld();

        interactionsHelper.init(config, object, K3D,
            pointsCallback, pointsIntersect.prepareGeometry(pointsGeometry), pointsIntersect.Intersect);

        for (i = 0; i < positions.length / 3; i++) {
            const s = (sizes && sizes[i]) || 1.0;

            object.setMatrixAt(
                i,
                (new THREE.Matrix4())
                    .identity()
                    .setPosition(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2])
                    .scale(new THREE.Vector3(s, s, s)),
            );
        }

        return Promise.resolve(object);
    },

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        if (typeof (changes.positions) !== 'undefined' && !changes.positions.timeSeries
            && changes.positions.data.length / 3 === obj.instanceMatrix.count) {
            const positions = changes.positions.data;

            for (let i = 0; i < positions.length / 3; i++) {
                const s = (config.sizes && config.sizes[i]) || 1.0;

                obj.setMatrixAt(
                    i,
                    (new THREE.Matrix4())
                        .identity()
                        .setPosition(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2])
                        .scale(new THREE.Vector3(s, s, s)),
                );
            }

            if (obj.interactions) {
                obj.stopInteraction();

                let pointsGeometry = new THREE.BufferGeometry();
                pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                interactionsHelper.init(config, obj, K3D,
                    pointsCallback, pointsIntersect.prepareGeometry(pointsGeometry), pointsIntersect.Intersect);
            }

            obj.instanceMatrix.needsUpdate = true;
            resolvedChanges.positions = null;
        }

        if (((typeof (changes.color_map) !== 'undefined' && !changes.color_map.timeSeries)
                || (typeof (changes.opacity_function) !== 'undefined' && !changes.opacity_function.timeSeries))
            && obj.geometry.attributes.attributes) {
            const canvas = colorMapHelper.createCanvasGradient(
                (changes.color_map && changes.color_map.data) || config.color_map.data,
                1024,
                1,
                (changes.opacity_function && changes.opacity_function.data) || config.opacity_function.data,
            );

            obj.material.uniforms.colormap.value.image = canvas;
            obj.material.uniforms.colormap.value.needsUpdate = true;

            resolvedChanges.color_map = null;
            resolvedChanges.opacity_function = null;
        }

        if (typeof (changes.color_range) !== 'undefined' && !changes.color_range.timeSeries
            && obj.geometry.attributes.attributes) {
            obj.material.uniforms.low.value = changes.color_range[0];
            obj.material.uniforms.high.value = changes.color_range[1];

            resolvedChanges.color_range = null;
        }

        if (typeof (changes.attribute) !== 'undefined' && !changes.attribute.timeSeries
            && obj.geometry.attributes.attributes
            && changes.attribute.data.length === obj.geometry.attributes.attributes.array.length) {
            obj.geometry.attributes.attributes.array.set(changes.attribute.data);
            obj.geometry.attributes.attributes.needsUpdate = true;

            resolvedChanges.attribute = null;
        }

        interactionsHelper.update(config, changes, resolvedChanges, obj);

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};
