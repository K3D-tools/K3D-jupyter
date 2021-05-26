function resetCameraGUI(gui, K3D) {
    const obj = {
        resetCamera() {
            K3D.getWorld().setCameraToFitScene(true);
            K3D.getWorld().render();
        },
    };

    gui.add(obj, 'resetCamera').name('Reset camera');
}

module.exports = resetCameraGUI;
