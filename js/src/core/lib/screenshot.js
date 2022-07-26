const FileSaver = require('file-saver');
const rasterizeHTML = require('rasterizehtml');

function getScreenshot(K3D, scale, onlyCanvas) {
    return new Promise((resolve, reject) => {
        const finalCanvas = document.createElement('canvas');
        const finalCanvasCtx = finalCanvas.getContext('2d');
        const htmlElementCanvas = document.createElement('canvas');
        const { clearColor } = K3D.parameters;
        const world = K3D.getWorld();
        const canvas3d = world.renderer.domElement;
        let renderPromise;
        let t;

        t = new Date().getTime();
        finalCanvas.width = Math.floor(canvas3d.width * scale);
        htmlElementCanvas.width = Math.floor(canvas3d.width * scale);
        finalCanvas.height = Math.floor(canvas3d.height * scale);
        htmlElementCanvas.height = Math.floor(canvas3d.height * scale);

        K3D.heavyOperationAsync = true;
        K3D.labels = [];
        K3D.dispatch(K3D.events.BEFORE_RENDER);

        if (clearColor >= 0) {
            finalCanvasCtx.fillStyle = `rgb(${
                (clearColor & 0xff0000) >> 16},${
                (clearColor & 0x00ff00) >> 8},${
                clearColor & 0x0000ff})`;

            finalCanvasCtx.fillRect(0, 0, finalCanvas.width, finalCanvas.height);
        }

        if (onlyCanvas) {
            renderPromise = Promise.resolve();
        } else {
            const styles = document.getElementsByTagName('style');
            let style = '';

            for (let i = 0; i < styles.length; i++) {
                style += styles[i].outerHTML;
            }
            renderPromise = rasterizeHTML.drawHTML(
                `<style>body{margin:0;}</style>
                ${style}
                ${world.overlayDOMNode.outerHTML}
                ${K3D.colorMapNode ? K3D.colorMapNode.outerHTML : ''}`,
                htmlElementCanvas,
                {
                    zoom: scale,
                },
            );
        }

        K3D.heavyOperationAsync = true;
        renderPromise.then((result) => {
            console.log(`K3D: Screenshot [html]: ${(new Date().getTime() - t) / 1000}`, 's');
            t = new Date().getTime();

            world.renderOffScreen(finalCanvas.width, finalCanvas.height).then((arrays) => {
                console.log(`K3D: Screenshot [canvas]: ${(new Date().getTime() - t) / 1000}`, 's');
                t = new Date().getTime();

                finalCanvasCtx.scale(1, -1);

                arrays.forEach((array) => {
                    const imageData = new ImageData(array, finalCanvas.width, finalCanvas.height);
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');

                    canvas.width = imageData.width;
                    canvas.height = imageData.height;
                    ctx.putImageData(imageData, 0, 0);

                    finalCanvasCtx.drawImage(canvas, 0, 0, finalCanvas.width, -finalCanvas.height);
                });

                if (result) {
                    finalCanvasCtx.scale(1, -1);
                    finalCanvasCtx.drawImage(htmlElementCanvas, 0, 0);
                }

                K3D.heavyOperationAsync = false;
                resolve(finalCanvas);
            });
        }, (e) => {
            reject(e);
        });
    });
}

function screenshotGUI(gui, K3D) {
    const obj = {
        screenshot() {
            getScreenshot(K3D, K3D.parameters.screenshotScale).then((canvas) => {
                canvas.toBlob((blob) => {
                    let filename = `K3D-${Date.now()}.png`;

                    if (K3D.parameters.name) {
                        filename = `${K3D.parameters.name}.png`;
                    }

                    FileSaver.saveAs(blob, filename);
                });
            }, () => {
                console.error('Failed to render screenshot.');
            });
        },
    };

    gui.add(obj, 'screenshot').name('Screenshot');
}

module.exports = {
    screenshotGUI,
    getScreenshot,
};
