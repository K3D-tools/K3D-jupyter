// Sole importer of three-gpu-pathtracer; the boundary is plain three.js - a Scene of
// Mesh(Standard|Physical)Material and a camera in, sample counts out.
const { WebGLPathTracer } = require('three-gpu-pathtracer');
const {
    BlueNoiseGenerator,
} = require('three-gpu-pathtracer/src/textures/blueNoise/BlueNoiseGenerator.js');

// the library's own stable-noise LCG (GCC constants), matched exactly
function lcgRandom(seed) {
    let state = seed;

    return function () {
        state = (1103515245 * state + 12345) % 0x80000000;

        return state / (0x80000000 - 1);
    };
}

// stableNoise pins the per-sample sequences only; the per-pixel offsets in
// stratifiedOffsetTexture are Math.random blue noise, so repeatability across page
// loads requires reseeding them too.
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

        // null when the tracer can run, else the reason - cinematic never silently rasterises
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
            tracer.dynamicLowRes = false;
            tracer.minSamples = 1;
            // the sample loop is driven externally
            tracer.renderDelay = 0;
            tracer.fadeDuration = 0;
            // per-sample seeds, not wall-clock: N samples are a pure function of the scene
            tracer.stableNoise = true;
            // K3D tone-maps the accumulation target itself; skip the library's canvas copy
            tracer.renderToCanvas = false;
            reseedOffsetTexture(tracer);

            if (typeof window !== 'undefined') {
                // diagnostic handle for headless probes
                window.__k3dTracer = tracer;
            }
        },

        setBounces(bounces) {
            tracer.bounces = bounces;
        },

        // Widens a glossy lobe in proportion to the roughness already accumulated along the
        // path, so a sharp specular seen directly stays sharp while the paths that produce
        // fireflies - rough bounce, then a mirror catching a small bright light - do not.
        setGlossyFilter(factor) {
            tracer.filterGlossyFactor = factor;
        },

        // One renderSample() advances one tile, bounding the GPU work per call: an
        // uninterrupted full-frame trace stalls the page and can trip the driver watchdog,
        // which tears down the GL context. Sizing by pixels holds that bound at any resolution.
        setTiles(width, height) {
            const perTile = 120000;
            const tiles = Math.min(6, Math.max(1, Math.ceil(Math.sqrt((width * height) / perTile))));

            tracer.tiles.set(tiles, tiles);

            return tiles * tiles;
        },

        // one sample is spread over this many renderSample() calls
        tileCount() {
            return Math.max(1, tracer.tiles.x) * Math.max(1, tracer.tiles.y);
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

        // exact pixel size for screenshots; renderScale only lands within +-1px of the target
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
