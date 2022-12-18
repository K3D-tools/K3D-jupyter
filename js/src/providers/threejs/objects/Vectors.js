const THREE = require('three');
const BufferGeometryUtils = require('three/examples/jsm/utils/BufferGeometryUtils');
const buffer = require('../../../core/lib/helpers/buffer');
const MeshLine = require('../helpers/THREE.MeshLine')(THREE);
const {getTwoColorsArray} = require('../helpers/Fn');
const {generateArrow} = require('../helpers/Fn');
const Text = require('./Text');
const Fn = require('../helpers/Fn');

const {commonUpdate} = Fn;
const {areAllChangesResolve} = Fn;

/**
 * Loader strategy to handle Vectors  object
 * @method Vector
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @param {K3D}
 */
module.exports = {
    create(config, K3D) {
        config.visible = typeof (config.visible) !== 'undefined' ? config.visible : true;
        config.origin_color = typeof (config.origin_color) !== 'undefined' ? config.origin_color : 255;
        config.head_color = typeof (config.head_color) !== 'undefined' ? config.head_color : 255;
        config.use_head = typeof (config.use_head) !== 'undefined' ? config.use_head : true;
        config.head_size = config.head_size || 1.0;
        config.line_width = config.line_width || 0.01;

        const modelMatrix = new THREE.Matrix4();
        const originColor = new THREE.Color(config.origin_color);
        const headColor = new THREE.Color(config.head_color);
        const vectors = config.vectors.data;
        const origins = config.origins.data;
        let colors = (config.colors && config.colors.data) || [];
        const useHead = config.use_head;
        const headSize = config.head_size;
        const object = new THREE.Group();
        let origin;
        let destination;
        let i;
        const labelSize = config.label_size;
        const {labels} = config;
        const labelsObjects = [];
        let heads = null;
        const lineVertices = [];

        colors = colors.length > 0 ? buffer.colorsToFloat32Array(colors)
            : getTwoColorsArray(originColor, headColor, (vectors.length / 3) * 2);

        if (vectors.length !== origins.length) {
            throw new Error('vectors and origins should have the same length');
        }

        if (colors && colors.length / 2 !== vectors.length) {
            throw new Error('there should be 2 colors for each vector');
        }

        const singleConeGeometry = new THREE.CylinderGeometry(
            0,
            0.025 * headSize,
            0.2 * headSize,
            5,
            1,
        ).translate(0, -0.1 * headSize, 0);

        for (i = 0; i < vectors.length; i += 3) {
            origin = new THREE.Vector3(origins[i], origins[i + 1], origins[i + 2]);
            destination = new THREE.Vector3(vectors[i], vectors[i + 1], vectors[i + 2]).add(origin);

            heads = generateArrow(
                useHead ? singleConeGeometry : null,
                lineVertices,
                heads,
                origin,
                destination,
                new THREE.Color(colors[i * 2 + 3], colors[i * 2 + 4], colors[i * 2 + 5]),
                0.2 * headSize,
            );

            if (labels) {
                if (labels[i / 3]) {
                    labelsObjects.push(
                        createText(labels[i / 3], origin, destination, labelSize, K3D),
                    );
                }
            }
        }

        if (useHead) {
            addHeads(heads, object);
        }

        let line = new MeshLine.MeshLine();
        const material = new MeshLine.MeshLineMaterial({
            color: new THREE.Color(1, 1, 1),
            opacity: 1.0,
            sizeAttenuation: true,
            transparent: true,
            lineWidth: config.line_width,
            resolution: new THREE.Vector2(K3D.getWorld().width, K3D.getWorld().height),
            side: THREE.DoubleSide,
        });

        line.setGeometry(new Float32Array(lineVertices), true, null, colors);
        line.geometry.computeBoundingSphere();
        line.geometry.computeBoundingBox();

        line = new THREE.Mesh(line.geometry, material);
        object.add(line);

        if (config.model_matrix) {
            modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
            object.applyMatrix4(modelMatrix);
        }

        object.updateMatrixWorld();

        const resizelistenerId = K3D.on(K3D.events.RESIZED, () => {
            line.material.uniforms.resolution.value.x = K3D.getWorld().width;
            line.material.uniforms.resolution.value.y = K3D.getWorld().height;
        });

        return Promise.all(labelsObjects).then((texts) => {
            texts.forEach((text) => {
                object.add(text);
            });

            object.onRemove = function () {
                texts.forEach((text) => {
                    text.onRemove();
                });

                K3D.off(K3D.events.RESIZED, resizelistenerId);
            };

            return object;
        });
    },

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj});
        }
        return false;
    },
};

function addHeads(heads, object) {
    heads = BufferGeometryUtils.mergeBufferGeometries(heads);
    heads.computeBoundingSphere();
    heads.computeBoundingBox();

    object.add(
        new THREE.Mesh(
            heads,
            new THREE.MeshBasicMaterial({vertexColors: THREE.VertexColors}),
        ),
    );
}

function createText(text, origin, destination, labelSize, K3D) {
    const center = origin.clone().add(destination).divideScalar(2);
    const textConfig = {
        position: [center.x, center.y, center.z],
        referencePoint: 'cb',
        text,
        size: labelSize,
    };

    return Text.create(textConfig, K3D);
}
