'use strict';

var colorsToFloat32Array = require('./../../../core/lib/helpers/buffer').colorsToFloat32Array,
    MeshLine = require('./../helpers/THREE.MeshLine'),
    Fn = require('./../helpers/Fn'),
    lut = require('./../../../core/lib/helpers/lut'),
    getColorsArray = Fn.getColorsArray;

/**
 * Loader strategy to handle Line object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config, K3D) {
    config.width = typeof(config.width) !== 'undefined' ? config.width : 0.1;

    var material = new MeshLine.MeshLineMaterial({
            color: new THREE.Color(1, 1, 1),
            opacity: 1.0,
            sizeAttenuation: true,
            transparent: true,
            lineWidth: config.width,
            resolution: new THREE.Vector2(K3D.getWorld().width, K3D.getWorld().height),
            side: THREE.DoubleSide
        }),
        verticesColors = (config.colors && config.colors.data) || null,
        color = new THREE.Color(config.color),
        colors = null,
        uvs = null,
        colorRange = config.color_range,
        colorMap = (config.color_map && config.color_map.data) || null,
        attribute = (config.attribute && config.attribute.data) || null,
        object,
        resizelistenerId,
        modelMatrix = new THREE.Matrix4(),
        position = config.vertices.data;

    if (attribute && colorRange && colorMap && attribute.length > 0 && colorRange.length > 0 && colorMap.length > 0) {
        var canvas = lut(colorMap, 1024);
        var texture = new THREE.CanvasTexture(canvas, THREE.UVMapping, THREE.ClampToEdgeWrapping,
            THREE.ClampToEdgeWrapping, THREE.NearestFilter, THREE.NearestFilter);
        texture.needsUpdate = true;

        material.uniforms.useMap.value = 1.0;
        material.uniforms.map.value = texture;

        uvs = new Float32Array(attribute.length);

        for (var i = 0; i < attribute.length; i++) {
            uvs[i] = (attribute[i] - colorRange[0]) / (colorRange[1] - colorRange[0]);
        }
    } else {
        colors = (verticesColors && verticesColors.length === position.length / 3 ?
                colorsToFloat32Array(verticesColors) : getColorsArray(color, position.length / 3)
        );
    }

    var line = new MeshLine.MeshLine();

    line.setGeometry(new Float32Array(position), false, null, colors, uvs);
    line.geometry.computeBoundingSphere();
    line.geometry.computeBoundingBox();

    object = new THREE.Mesh(line.geometry, material);
    modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
    object.applyMatrix(modelMatrix);

    object.updateMatrixWorld();

    resizelistenerId = K3D.on(K3D.events.RESIZED, function () {
        // update outlines
        object.material.uniforms.resolution.value.x = K3D.getWorld().width;
        object.material.uniforms.resolution.value.y = K3D.getWorld().height;
    });

    object.onRemove = function () {
        K3D.off(K3D.events.RESIZED, resizelistenerId);
        if (object.material.uniforms.map.value) {
            object.material.uniforms.map.value.dispose();
            object.material.uniforms.map.value = undefined;
        }
    };

    return Promise.resolve(object);
};
