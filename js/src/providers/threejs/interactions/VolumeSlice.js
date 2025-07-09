const THREE = require('three');
const { decodeFloat16 } = require('../../../core/lib/helpers/math');

/**
 * Interactions handlers for VolumeSlice object
 * @memberof K3D.Providers.ThreeJS.Interactions
 */

function prepareParam(param) {
    const texture = param.object.material.uniforms.volumeTexture.value[0];
    const isFloat16 = texture.type === THREE.HalfFloatType;
    const volumeData = texture.source.data;

    const origin = param.object.position.clone().sub(param.object.scale.clone().divideScalar(2));
    const coord = param.point.clone().sub(origin).divide(param.object.scale).multiply(
        new THREE.Vector3(volumeData.width - 1, volumeData.height - 1, volumeData.depth - 1),
    ).round();

    let value = volumeData.data[coord.z * volumeData.width * volumeData.height
    + coord.y * volumeData.width
    + coord.x];

    if (isFloat16) {
        value = decodeFloat16(value);
    }

    return {
        position: param.point.toArray(),
        normal: param.face.normal.toArray(),
        distance: param.distance,
        face_index: param.faceIndex,
        face: [param.face.a, param.face.b, param.face.c],
        uv: param.uv,
        coord: [coord.x, coord.y, coord.z],
        K3DIdentifier: param.object.K3DIdentifier,
        value
    };
}

module.exports = function (object, K3D) {
    return {
        onHover(intersect) {
            K3D.dispatch(K3D.events.OBJECT_HOVERED, prepareParam(intersect));
            return false;
        },
        onClick(intersect) {
            K3D.dispatch(K3D.events.OBJECT_CLICKED, prepareParam(intersect));
            return false;
        },
    };
};
