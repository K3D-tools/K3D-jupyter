'use strict';

// These jitter vectors are specified in integers because it is easier.
// I am assuming a [-8,8) integer grid, but it needs to be mapped onto [-0.5,0.5)
// before being used, thus these integers need to be scaled by 1/16.
//
// Sample patterns reference: https://msdn.microsoft.com/en-us/library/windows/desktop/ff476218%28v=vs.85%29.aspx?f=255&MSPPError=-2147217396

var JitterVectors = [
    [
        [0, 0]
    ],
    [
        [4, 4], [-4, -4]
    ],
    [
        [-2, -6], [6, -2], [-6, 2], [2, 6]
    ],
    [
        [1, -3], [-1, 3], [5, 1], [-3, -5],
        [-5, 5], [-7, -1], [3, 7], [7, -7]
    ],
    [
        [1, 1], [-1, -3], [-3, 2], [4, -1],
        [-5, -2], [2, 5], [5, 3], [3, -5],
        [-2, 6], [0, -7], [-4, -6], [-6, 4],
        [-8, 0], [7, -4], [6, 7], [-7, -8]
    ],
    [
        [-4, -7], [-7, -5], [-3, -5], [-5, -4],
        [-1, -4], [-2, -2], [-6, -1], [-4, 0],
        [-7, 1], [-1, 2], [-6, 3], [-3, 3],
        [-7, 6], [-3, 6], [-5, 7], [-1, 7],
        [5, -7], [1, -6], [6, -5], [4, -4],
        [2, -3], [7, -2], [1, -1], [4, -1],
        [2, 1], [6, 2], [0, 4], [4, 4],
        [2, 5], [7, 5], [5, 6], [3, 7]
    ]
];

function getArrayFromRenderTarget(renderer, rt) {
    var array = new Uint8Array(rt.width * rt.height * 4);

    renderer.readRenderTargetPixels(rt, 0, 0, rt.width, rt.height, array);
    return new Uint8ClampedArray(array, rt.width, rt.height);
}

module.exports = function (renderer, scene, camera, rt, fullWidth, fullHeight, chunk_heights, sampleLevel) {
    return new Promise(function (resolve) {

        var jitterOffsets = JitterVectors[Math.max(0, Math.min(sampleLevel, 5))];
        var autoClear = renderer.autoClear;
        var copyShader = THREE.CopyShader;
        var copyUniforms = THREE.UniformsUtils.clone(copyShader.uniforms);
        var copyMaterial = new THREE.ShaderMaterial({
            uniforms: copyUniforms,
            vertexShader: copyShader.vertexShader,
            fragmentShader: copyShader.fragmentShader,
            premultipliedAlpha: true,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthTest: false,
            depthWrite: false
        });
        var sampleRenderTarget = new THREE.WebGLRenderTarget(rt.width, rt.height, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format: THREE.RGBAFormat
        });

        var camera2 = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        var scene2 = new THREE.Scene();
        var quad2 = new THREE.Mesh(new THREE.PlaneBufferGeometry(2, 2), copyMaterial);

        quad2.frustumCulled = false;
        scene2.add(quad2);

        renderer.autoClear = false;

        var oldClearColor = renderer.getClearColor().getHex();
        var oldClearAlpha = renderer.getClearAlpha();

        var baseSampleWeight = 1.0 / jitterOffsets.length;
        var roundingRange = 1 / 32;

        copyUniforms.tDiffuse.value = sampleRenderTarget.texture;

        var fullresImage = new Uint8ClampedArray(fullWidth * fullHeight * 4);
        var p = Promise.resolve();

        chunk_heights.forEach(function (c) {
            p = p.then(function () {
                var width = rt.width, height = c[1];

                for (var i = 0; i < jitterOffsets.length; i++) {
                    var jitterOffset = jitterOffsets[i];

                    camera.setViewOffset(fullWidth, fullHeight,
                        jitterOffset[0] * 0.0625, jitterOffset[1] * 0.0625 + c[0], // 0.0625 = 1 / 16
                        width, height);

                    var sampleWeight = baseSampleWeight;
                    var uniformCenteredDistribution = (-0.5 + (i + 0.5) / jitterOffsets.length);
                    sampleWeight += roundingRange * uniformCenteredDistribution;

                    copyUniforms.opacity.value = sampleWeight;
                    renderer.setClearColor(0x000000, 0);
                    renderer.setRenderTarget(sampleRenderTarget);
                    renderer.clear();
                    renderer.render(scene, camera);

                    renderer.setRenderTarget(rt);

                    if (i === 0) {
                        renderer.setClearColor(0x000000, 0.0);
                        renderer.clear();
                    }

                    renderer.render(scene2, camera2);
                }

                fullresImage.set(
                    getArrayFromRenderTarget(renderer, rt).subarray(0, width * height * 4),
                    (fullHeight - height - c[0]) * width * 4
                );
            });

            p = p.then(function () {
                return new Promise(function (resolve) {
                    setTimeout(resolve, 100);
                });
            });
        });

        p = p.then(function () {
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
