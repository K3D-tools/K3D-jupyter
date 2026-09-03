// Sole importer of three-gpu-pathtracer; the boundary is plain three.js - a Scene of
// Mesh(Standard|Physical)Material and a camera in, sample counts out.
const THREE = require('three');
const { WebGLPathTracer } = require('three-gpu-pathtracer');
const {
    BlueNoiseGenerator,
} = require('three-gpu-pathtracer/src/textures/blueNoise/BlueNoiseGenerator.js');
const {
    GenerateMeshBVHWorker,
} = require('three-mesh-bvh/src/workers/GenerateMeshBVHWorker.js');
const { WorkerBase } = require('three-mesh-bvh/src/workers/utils/WorkerBase.js');
const bvhWorkerSource = require('../../../../core/lib/bvhWorkerSource');

// GenerateMeshBVHWorker builds its Worker from a URL relative to the module, which neither
// bundle can resolve; requiring it is still what makes webpack emit the worker chunk, and its
// client half of the protocol is reused verbatim over a worker built from the chunk's source.
class BlobBVHWorker extends WorkerBase {
    constructor(source) {
        const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));

        super(new Worker(url));
        URL.revokeObjectURL(url);

        this.name = 'GenerateMeshBVHWorker';
    }
}

BlobBVHWorker.prototype.runTask = GenerateMeshBVHWorker.prototype.runTask;

// the library's own stable-noise LCG (GCC constants), matched exactly
function lcgRandom(seed) {
    let state = seed;

    return function () {
        state = (1103515245 * state + 12345) % 0x80000000;

        return state / (0x80000000 - 1);
    };
}

// stableNoise pins the per-sample sequences only; the per-pixel offsets in
// stratifiedOffsetTexture are blue noise drawn from Math.random, so a repeatable
// render needs them drawn from the seed too. null hands them back to Math.random.
function reseedOffsetTexture(tracer, seed) {
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
    generator.random = seed === null ? Math.random : lcgRandom(seed);

    const { data, maxValue } = generator.generate();
    const pixels = texture.image.data;

    for (let i = 0; i < data.length; i++) {
        pixels[i] = data[i] / maxValue;
    }

    texture.needsUpdate = true;
}

module.exports = function createWebGLBackend(renderer) {
    let tracer = null;
    let workerProbe = null;

    // Resolves with a usable worker or null: availability is decided by building one triangle
    // rather than assumed, since the source has to reach here from the page or the kernel. The
    // parallel worker is not used - it needs SharedArrayBuffer, which requires cross-origin
    // isolation no notebook server sends.
    function probeBVHWorker() {
        if (workerProbe !== null) {
            return workerProbe;
        }

        workerProbe = bvhWorkerSource.read().then((source) => {
            if (source === null) {
                // a busy kernel answers late as readily as a missing chunk answers never:
                // forget this attempt so the next build asks again
                workerProbe = null;

                return null;
            }

            return new Promise((resolve) => {
                let worker = null;

                function fail() {
                    if (worker !== null) {
                        try {
                            worker.dispose();
                        } catch (e) {
                            // already dead
                        }
                    }

                    resolve(null);
                }

                try {
                    worker = new BlobBVHWorker(source);
                } catch (e) {
                    fail();

                    return;
                }

                const geometry = new THREE.BufferGeometry();
                const triangle = new THREE.BufferAttribute(new Float32Array(9), 3);

                geometry.setAttribute('position', triangle);

                // in this same tick, so the task's own handler replaces WorkerBase's, which
                // reports a worker that cannot start by throwing out of the event listener
                worker.generate(geometry).then(() => resolve(worker), fail);
            });
        });

        return workerProbe;
    }

    // An off-thread build leaves the GPU copy of the per-triangle material index behind the
    // merged geometry: while every material still looks alike the frame is right, and the
    // first opacity or roughness edit afterwards shades triangles from the wrong material.
    // Re-uploading it from the geometry the generator merged is what the synchronous build
    // effectively gets for free.
    function refreshMaterialIndex() {
        const generator = tracer._generator;
        const material = tracer._pathTracer && tracer._pathTracer.material;
        const attribute = generator && generator.geometry
            && generator.geometry.attributes.materialIndex;

        if (!material || !material.materialIndexAttribute || !attribute) {
            throw new Error(
                'cinematic: three-gpu-pathtracer internals changed - cannot refresh the '
                + 'material index after an off-thread build',
            );
        }

        material.materialIndexAttribute.updateFrom(attribute);
    }

    // the worker is an OS thread outliving the widget that spawned it; a re-run cell must
    // not leave one behind
    function releaseBVHWorker() {
        const probe = workerProbe;

        workerProbe = null;

        if (probe !== null) {
            probe.then((worker) => {
                if (worker !== null) {
                    worker.dispose();
                }
            });
        }
    }

    // The tracer's merge appends one group per source geometry and never clears them, so a
    // rebuilt scene stacks a fresh set on top of every previous one. Harmless for the image -
    // the duplicates repeat the same ranges - but it grows for the life of the page.
    function clearMergedGroups() {
        const geometry = tracer._generator && tracer._generator.geometry;

        if (geometry) {
            geometry.clearGroups();
        }
    }

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
            // noise is pinned or freed through setSeed(), driven by the cinematicSeed parameter
            // K3D tone-maps the accumulation target itself; skip the library's canvas copy
            tracer.renderToCanvas = false;
            // the generator asks for one primitive per leaf through maxLeafTris, which is
            // deprecated and warns on every rebuild; bvhOptions is spread last, so saying the
            // same thing under its current name and clearing the old key silences it
            tracer._generator.bvhOptions = { maxLeafTris: undefined, targetLeafSize: 1 };

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

        // null: the library takes its own Math.random paths, so every accumulation differs.
        // An integer: seeded stratified jitter plus seeded offsets, so N samples are a pure
        // function of the scene - what a reference-image suite needs and a notebook does not.
        setSeed(seed) {
            const pinned = seed !== null && seed !== undefined;

            tracer.stableNoise = pinned;
            reseedOffsetTexture(tracer, pinned ? seed : null);
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
            clearMergedGroups();
            tracer.setScene(scene, camera);
        },

        // Resolves true when the BVH was built off the main thread, false when no worker could
        // run and the caller has to build it synchronously. A rejection leaves the generator
        // holding the failed build, so the worker is dropped and the retry stays on the main
        // thread.
        setSceneAsync(scene, camera, onProgress) {
            return probeBVHWorker().then((worker) => {
                if (worker === null) {
                    return false;
                }

                tracer.setBVHWorker(worker);
                clearMergedGroups();

                return tracer.setSceneAsync(scene, camera, { onProgress }).then(
                    () => true,
                    (e) => {
                        workerProbe = Promise.resolve(null);

                        throw e;
                    },
                );
            });
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
            refreshMaterialIndex();
        },

        renderSample() {
            tracer.renderSample();

            return { samples: tracer.samples };
        },

        reset() {
            tracer.reset();
        },

        releaseBVHWorker,

        dispose() {
            releaseBVHWorker();

            if (tracer) {
                tracer.dispose();
                tracer = null;
            }
        },
    };
};
