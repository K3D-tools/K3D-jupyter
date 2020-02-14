'use strict';

var THREE = require('three'),
    buffer = require('./../../../core/lib/helpers/buffer'),
    Fn = require('./../helpers/Fn'),
    areAllChangesResolve = Fn.areAllChangesResolve,
    commonUpdate = Fn.commonUpdate,
    getColorsArray = Fn.getColorsArray;

/**
 * Loader strategy to handle Points object
 * @method PointsMesh
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configuration params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config) {
        var modelMatrix = new THREE.Matrix4(),
            color = new THREE.Color(config.color),
            positions = config.positions.data,
            pointColors = (config.colors && config.colors.data) || null,
            meshDetail = typeof (config.mesh_detail) !== 'undefined' ? config.mesh_detail : 2,
            colors,
            opacities = (config.opacities && config.opacities.data &&
                         config.opacities.data.length === positions.length / 3) ? config.opacities.data : null,
            object,
            colorsToFloat32Array = buffer.colorsToFloat32Array,
            phongShader = THREE.ShaderLib.phong,
            material = new THREE.ShaderMaterial({
                uniforms: THREE.UniformsUtils.merge([phongShader.uniforms, {
                    shininess: {value: 50},
                    opacity: {value: config.opacity},
                }]),
                defines: {
                    USE_PER_POINT_OPACITY: (opacities !== null ? 1 : 0)
                },
                vertexShader: require('./shaders/PointsMesh.vertex.glsl'),
                fragmentShader: require('./shaders/PointsMesh.fragment.glsl'),
                depthTest: (config.opacity === 1.0 && opacities === null),
                transparent: (config.opacity !== 1.0 || opacities !== null),
                lights: true,
                clipping: true,
                vertexColors: THREE.VertexColors
            }),
            sphereGeometry = new THREE.IcosahedronBufferGeometry(config.point_size * 0.5, meshDetail),
            instancedGeometry = new THREE.InstancedBufferGeometry().copy(sphereGeometry),
            geometry = new THREE.BufferGeometry();

        colors = (pointColors && pointColors.length === positions.length / 3 ?
                colorsToFloat32Array(pointColors) : getColorsArray(color, positions.length / 3)
        );

        instancedGeometry.setAttribute('offset', new THREE.InstancedBufferAttribute(new Float32Array(positions), 3));
        instancedGeometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array(colors), 3));

        if (opacities) {
            instancedGeometry.setAttribute('opacities',
                new THREE.InstancedBufferAttribute(opacities, 1).setUsage(THREE.DynamicDrawUsage));
        }

        // boundingBox & boundingSphere
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.computeBoundingSphere();
        instancedGeometry.boundingSphere = geometry.boundingSphere.clone();
        geometry.computeBoundingBox();
        instancedGeometry.boundingBox = geometry.boundingBox.clone();
        Fn.expandBoundingBox(instancedGeometry.boundingBox, config.point_size * 0.5);

        object = new THREE.Mesh(instancedGeometry, material);

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },

    update: function (config, changes, obj) {
        var resolvedChanges = {};

        // if (typeof(changes.opacity) !== 'undefined' && !changes.opacity.timeSeries) {
        //     obj.material.uniforms.opacity.value = changes.opacity;
        //
        //     resolvedChanges.opacity = null;
        // }

        commonUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
};
