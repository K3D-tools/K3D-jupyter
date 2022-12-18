// These jitter vectors are specified in integers because it is easier.
// I am assuming a [-8,8) integer grid, but it needs to be mapped onto [-0.5,0.5)
// before being used, thus these integers need to be scaled by 1/16.
//
// Sample patterns reference:
// https://msdn.microsoft.com/en-us/library/windows/desktop/ff476218%28v=vs.85%29.aspx?f=255&MSPPError=-2147217396

const THREE = require('three');

const JitterVectors = [
    [
        [0, 0],
    ],
    [
        [4, 4], [-4, -4],
    ],
    [
        [-2, -6], [6, -2], [-6, 2], [2, 6],
    ],
    [
        [1, -3], [-1, 3], [5, 1], [-3, -5],
        [-5, 5], [-7, -1], [3, 7], [7, -7],
    ],
    [
        [1, 1], [-1, -3], [-3, 2], [4, -1],
        [-5, -2], [2, 5], [5, 3], [3, -5],
        [-2, 6], [0, -7], [-4, -6], [-6, 4],
        [-8, 0], [7, -4], [6, 7], [-7, -8],
    ],
    [
        [-4, -7], [-7, -5], [-3, -5], [-5, -4],
        [-1, -4], [-2, -2], [-6, -1], [-4, 0],
        [-7, 1], [-1, 2], [-6, 3], [-3, 3],
        [-7, 6], [-3, 6], [-5, 7], [-1, 7],
        [5, -7], [1, -6], [6, -5], [4, -4],
        [2, -3], [7, -2], [1, -1], [4, -1],
        [2, 1], [6, 2], [0, 4], [4, 4],
        [2, 5], [7, 5], [5, 6], [3, 7],
    ],
];

function getArrayFromRenderTarget(renderer, rt) {
    const array = new Float32Array(rt.width * rt.height * 4);
    const image = new Uint8ClampedArray(rt.width * rt.height * 4);
    let i;

    renderer.readRenderTargetPixels(rt, 0, 0, rt.width, rt.height, array);

    for (i = 0; i < array.length; i++) {
        image[i] = Math.floor(array[i] * 256.0);
    }

    return image;
}

module.exports = function (renderer, scene, camera, rt, fullWidth, fullHeight, chunkHeights, sampleLevel, render) {
    return new Promise((resolve) => {
        const jitterOffsets = JitterVectors[Math.max(0, Math.min(sampleLevel, 5))];
        const {autoClear} = renderer;
        const copyShader = THREE.CopyShader;
        const copyUniforms = THREE.UniformsUtils.clone(copyShader.uniforms);
        const copyMaterial = new THREE.ShaderMaterial({
            uniforms: copyUniforms,
            vertexShader: copyShader.vertexShader,
            fragmentShader: copyShader.fragmentShader,
            premultipliedAlpha: true,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthTest: false,
            depthWrite: false,
        });
        const sampleRenderTarget = new THREE.WebGLRenderTarget(rt.width, rt.height, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format: THREE.RGBAFormat,
            stencilBuffer: true,
        });

        const camera2 = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        const scene2 = new THREE.Scene();
        const quad2 = new THREE.Mesh(new THREE.PlaneBufferGeometry(2, 2), copyMaterial);

        quad2.frustumCulled = false;
        scene2.add(quad2);

        renderer.autoClear = false;

        let oldClearColor = new THREE.Color();
        renderer.getClearColor(oldClearColor);
        oldClearColor = oldClearColor.getHex();

        const oldClearAlpha = renderer.getClearAlpha();

        const baseSampleWeight = 1.0 / jitterOffsets.length;
        const roundingRange = 1 / 32;

        copyUniforms.tDiffuse.value = sampleRenderTarget.texture;

        const fullresImage = new Uint8ClampedArray(fullWidth * fullHeight * 4);
        let p = Promise.resolve();

        chunkHeights.forEach((c) => {
            p = p.then(() => {
                const {width} = rt;
                const height = c[1];

                for (let i = 0; i < jitterOffsets.length; i++) {
                    const jitterOffset = jitterOffsets[i];

                    camera.setViewOffset(
                        fullWidth,
                        fullHeight,
                        jitterOffset[0] * 0.0625,
                        jitterOffset[1] * 0.0625 + c[0], // 0.0625 = 1 / 16
                        width,
                        height,
                    );

                    let sampleWeight = baseSampleWeight;
                    const uniformCenteredDistribution = (-0.5 + (i + 0.5) / jitterOffsets.length);
                    sampleWeight += roundingRange * uniformCenteredDistribution;

                    copyUniforms.opacity.value = sampleWeight;

                    renderer.setClearColor(0x000000, 0);
                    renderer.setRenderTarget(sampleRenderTarget);
                    renderer.clear();
                    render(scene, camera, sampleRenderTarget);

                    renderer.setRenderTarget(rt);

                    if (i === 0) {
                        renderer.setClearColor(0x000000, 0.0);
                        renderer.clear();
                    }

                    renderer.render(scene2, camera2);
                }

                fullresImage.set(
                    getArrayFromRenderTarget(renderer, rt).subarray(0, width * height * 4),
                    (fullHeight - height - c[0]) * width * 4,
                );
            });

            p = p.then(() => new Promise((chunkResolve) => {
                setTimeout(chunkResolve, 100);
            }));
        });

        p = p.then(() => {
            if (camera.clearViewOffset) {
                camera.clearViewOffset();
            }

            renderer.autoClear = autoClear;
            renderer.setClearColor(oldClearColor, oldClearAlpha);
            sampleRenderTarget.dispose();

            resolve(fullresImage);
        });
    });
};
