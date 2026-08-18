// The ONLY module importing three-gpu-pathtracer (WebGPU-readiness rule #1 of
// renderer_cinematic.md). Everything crossing this boundary is plain three.js:
// a Scene of Mesh(Standard|Physical)Material and a camera in, sample counts out.
const { WebGLPathTracer } = require('three-gpu-pathtracer');
const {
    BlueNoiseGenerator,
} = require('three-gpu-pathtracer/src/textures/blueNoise/BlueNoiseGenerator.js');

// the library's own stable-noise LCG (GCC constants) - reused so every random
// stream in a cinematic render derives from the same scheme
function lcgRandom(seed) {
    let state = seed;

    return function () {
        state = (1103515245 * state + 12345) % 0x80000000;

        return state / (0x80000000 - 1);
    };
}

// stableNoise pins the per-sample sequences, but the per-pixel decorrelation
// offsets (stratifiedOffsetTexture) are blue noise rolled with Math.random at
// material construction and never reset - without reseeding them reference
// images differ between page loads even though a single page is repeatable
function reseedOffsetTexture(tracer) {
    const pathTracer = tracer._pathTracer;
    const texture = pathTracer && pathTracer.material
        && pathTracer.material.stratifiedOffsetTexture;

    if (!texture || !texture.image || !texture.image.data) {
        throw new Error(
            'cinematic: three-gpu-pathtracer internals changed - cannot pin the noise seed',
        );
    }

    const generator = new BlueNoiseGenerator();

    generator.size = texture.image.width;
    generator.random = lcgRandom(1);

    const { data, maxValue } = generator.generate();
    const pixels = texture.image.data;

    for (let i = 0; i < data.length; i++) {
        pixels[i] = data[i] / maxValue;
    }

    texture.needsUpdate = true;
}

module.exports = function createWebGLBackend(renderer) {
    let tracer = null;

    return {
        isSupported() {
            return this.unsupportedReason() === null;
        },

        // null when the path tracer can run; otherwise the concrete reason for
        // the no-fallback error overlay (decision #2 of renderer_cinematic.md)
        unsupportedReason() {
            try {
                const gl = renderer.getContext();

                if (!gl || !renderer.capabilities.isWebGL2) {
                    return 'WebGL2 is not available';
                }
                if (!gl.getExtension('EXT_color_buffer_float')) {
                    return 'renderable float textures (EXT_color_buffer_float) are not available';
                }

                return null;
            } catch (e) {
                return e.message || 'WebGL initialization failed';
            }
        },

        init() {
            tracer = new WebGLPathTracer(renderer);
            // headless determinism and throughput beat page responsiveness here;
            // the interactive loop revisits tiling in stage 3
            tracer.tiles.set(1, 1);
            tracer.dynamicLowRes = false;
            tracer.minSamples = 1;
            // the sample loop is driven externally - no cinematic fade-ins and
            // no grace delay before tracing starts
            tracer.renderDelay = 0;
            tracer.fadeDuration = 0;
            // per-sample seeds instead of wall-clock noise: the accumulation of
            // N samples is a pure function of the scene - references depend on it
            tracer.stableNoise = true;
            // presentation belongs to K3D: the accumulation target goes
            // through the shared tone-mapping blit, not the library's copy
            tracer.renderToCanvas = false;
            reseedOffsetTexture(tracer);
        },

        setBounces(bounces) {
            tracer.bounces = bounces;
        },

        setScene(scene, camera) {
            tracer.setScene(scene, camera);
        },

        updateCamera() {
            tracer.updateCamera();
        },

        updateEnvironment() {
            tracer.updateEnvironment();
        },

        // screenshots accumulate at the exact requested resolution instead of
        // the canvas size - renderScale would floor to +-1px of the target
        setFixedSize(width, height) {
            tracer.synchronizeRenderSize = false;
            tracer._pathTracer.setSize(width, height);
        },

        releaseFixedSize() {
            tracer.synchronizeRenderSize = true;
        },

        targetTexture() {
            const target = tracer.target;

            return target.texture || target;
        },

        updateMaterials() {
            tracer.updateMaterials();
        },

        renderSample() {
            tracer.renderSample();

            return { samples: tracer.samples };
        },

        reset() {
            tracer.reset();
        },

        dispose() {
            if (tracer) {
                tracer.dispose();
                tracer = null;
            }
        },
    };
};
