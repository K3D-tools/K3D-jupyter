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
        const { autoClear } = renderer;
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
        const quad2 = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), copyMaterial);

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

        let p = Promise.resolve();

        // Every chunk is rendered through the SAME projection an unchunked render uses, and
        // is selected by scissor instead. A per-chunk sub-frustum is a different projection
        // matrix - setViewOffset with a chunk height scales y by fullHeight/chunkHeight - and
        // float32 rasterisation then rounds a fraction of vertical positions onto a different
        // 1/256 subpixel, so the strips did not reassemble into the image a single pass draws.
        // Scissor changes nothing a fragment can observe: same matrix, same viewport, same
        // gl_FragCoord, fragments outside the rect discarded. The render targets carry the
        // rect rather than the renderer, because setRenderTarget copies scissor state off the
        // target it binds (three.module.js) and would undo it on every pass.
        chunkHeights.forEach((c) => {
            p = p.then(() => {
                const { width } = rt;
                const height = c[1];
                // readback and scissor are both bottom-up; c[0] counts from the top
                const bottom = fullHeight - c[0] - height;

                // one chunk covers the target, so the rect is the whole of it and the test is
                // semantically a no-op - but only semantically: a software rasteriser pays for
                // it per fragment, which is every pixel of every jitter pass
                const chunked = chunkHeights.length > 1;

                [rt, sampleRenderTarget].forEach((target) => {
                    target.scissor.set(0, bottom, width, height);
                    target.scissorTest = chunked;
                });

                for (let i = 0; i < jitterOffsets.length; i++) {
                    const jitterOffset = jitterOffsets[i];

                    camera.setViewOffset(
                        fullWidth,
                        fullHeight,
                        jitterOffset[0] * 0.0625,
                        jitterOffset[1] * 0.0625, // 0.0625 = 1 / 16
                        width,
                        fullHeight,
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

                [rt, sampleRenderTarget].forEach((target) => {
                    target.scissorTest = false;
                });
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

            // one readback of the whole target: each chunk was scissored into its own rows
            // of it, so there is nothing left to reassemble
            const image = getArrayFromRenderTarget(renderer, rt);

            sampleRenderTarget.dispose();
            copyMaterial.dispose();
            quad2.geometry.dispose();

            resolve(image);
        });
    });
};
