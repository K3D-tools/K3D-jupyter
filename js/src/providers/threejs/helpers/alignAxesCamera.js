const THREE = require('three');

// up before lookAt: the roll of the view is decided by the up vector, so setting it afterwards
// leaves the gizmo tilted until something aligns it again
module.exports = function alignAxesCamera(self, K3D) {
    const camDistance = (3.0 * 0.5) / Math.tan(THREE.MathUtils.degToRad(K3D.parameters.cameraFov / 2.0));

    self.axesHelper.camera.position.copy(
        self.camera.position.clone().sub(self.controls.target).normalize().multiplyScalar(camDistance),
    );
    self.axesHelper.camera.up.copy(self.camera.up);
    self.axesHelper.camera.lookAt(0, 0, 0);
    self.axesHelper.camera.updateMatrixWorld();
};
