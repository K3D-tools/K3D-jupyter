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
                    opacity: {value: config.opacity}
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
            i,
            boundingBoxGeometry = new THREE.BufferGeometry(),
            geometry = new THREE.IcosahedronBufferGeometry(config.point_size * 0.5, meshDetail);

        colors = (pointColors && pointColors.length === positions.length / 3 ?
                colorsToFloat32Array(pointColors) : getColorsArray(color, positions.length / 3)
        );

        geometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array(colors), 3));

        if (opacities) {
            geometry.setAttribute('opacities',
                new THREE.InstancedBufferAttribute(opacities, 1));
        }

        // boundingBox & boundingSphere
        boundingBoxGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        boundingBoxGeometry.computeBoundingSphere();
        boundingBoxGeometry.computeBoundingBox();
        Fn.expandBoundingBox(boundingBoxGeometry.boundingBox, config.point_size * 0.5);

        geometry.boundingBox = boundingBoxGeometry.boundingBox.clone();
        object = new THREE.InstancedMesh(geometry, material, positions.length / 3);
        object.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix4(modelMatrix);
        object.updateMatrixWorld();

        for (i = 0; i < positions.length / 3; i++) {
            object.setMatrixAt(i,
                (new THREE.Matrix4()).setPosition(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]));
        }

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
