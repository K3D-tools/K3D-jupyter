// Importer of the three-gpu-pathtracer runtime (sceneProxy borrows only its FogVolumeMaterial
// marker); the boundary is plain three.js - a Scene of Mesh(Standard|Physical)Material and a
// camera in, sample counts out.
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
const { VolumePathTracingMaterial } = require('./volume/VolumePathTracingMaterial');
const createVarianceBuffer = require('./variance');

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
    let volumeUnsupportedReported = false;
    let tracer = null;
    let workerProbe = null;
    let variance = null;
    // survives the buffer being switched off around a fixed-size render
    let wantVariance = false;
    // set when something consumes the buffer rather than merely reporting it, which is what
    // decides whether a fixed-size render may switch it off underneath
    let varianceRequired = false;
    // Bumped by everything that discards the accumulation. The sample counter alone
    // cannot stand in for it: with a single tile, a reset from sample 1 back to sample 1
    // leaves the counter where it was, and a measurement of two unrelated halves would
    // read as convergence.
    let epoch = 0;

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

    // The library builds its tracer material in the constructor and primes it with an empty
    // scene. The replacement has to land before the first real scene, since every upload
    // setScene makes goes to whatever material the quad holds then; and it has to be primed
    // the same way, because the generator reports an unchanged empty scene as no change and
    // would leave the BVH uniforms of a fresh material unset.
    function installVolumeMaterial() {
        const pathTracer = tracer._pathTracer;
        const generator = tracer._generator;
        const previous = pathTracer && pathTracer.material;

        if (!previous || !generator || typeof tracer._updateFromResults !== 'function') {
            throw new Error(
                'cinematic: three-gpu-pathtracer internals changed - cannot install the volume '
                + 'material',
            );
        }

        const material = new VolumePathTracingMaterial();

        pathTracer.material = material;

        if (pathTracer.material !== material) {
            throw new Error(
                'cinematic: three-gpu-pathtracer internals changed - the volume material did '
                + 'not reach the tracer',
            );
        }

        const results = generator.generate();

        results.bvhChanged = true;
        results.needsMaterialIndexUpdate = true;
        tracer._updateFromResults(tracer.scene, tracer.camera, results);
        previous.dispose();
    }

    // the medium the tracer material tracks through: the first proxy carrying a volume
    // description, or none
    function syncVolume(scene) {
        const material = tracer._pathTracer && tracer._pathTracer.material;

        if (!material || typeof material.setVolume !== 'function'
            || typeof material.clearVolume !== 'function') {
            throw new Error(
                'cinematic: three-gpu-pathtracer internals changed - the volume material is '
                + 'not installed',
            );
        }

        let volume = null;

        scene.traverse((node) => {
            if (volume === null && node.userData && node.userData.k3dVolume) {
                volume = node.userData.k3dVolume;
            }
        });

        if (volume !== null) {
            material.setVolume(volume, renderer);
        } else {
            material.clearVolume();
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

        // false on contexts short of texture units or uniform vectors; the volume then stays raster
        // the tracer is the state: a dispose from anywhere has to be visible to the caller,
        // or ensurePrepared would skip init() and every call would land on null
        isReady() {
            return tracer !== null;
        },

        volumeSupported() {
            const reason = VolumePathTracingMaterial.unsupportedReason(renderer);

            if (reason !== null && !volumeUnsupportedReported) {
                console.warn(`K3D.cinematic: ${reason}; volumes stay on the raster layer`);
                volumeUnsupportedReported = true;
            }

            return reason === null;
        },

        init() {
            tracer = new WebGLPathTracer(renderer);
            installVolumeMaterial();
            tracer.dynamicLowRes = false;
            tracer.minSamples = 1;
            // the sample loop is driven externally
            tracer.renderDelay = 0;
            tracer.fadeDuration = 0;
            // noise is pinned or freed through setSeed(), driven by the cinematicSeed parameter
            // K3D tone-maps the accumulation target itself; skip the library's canvas copy
            tracer.renderToCanvas = false;
            // created dormant: nothing is allocated until something asks to measure
            variance = createVarianceBuffer(renderer);
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
        // which tears down the GL context. A ceiling on the tile count breaks that bound at
        // exactly the resolutions that need it - at 6, a 4K frame gets tiles of 230k pixels,
        // twice the budget - so it sits where the per-call overhead of 256 tiles starts to
        // matter instead, which is past any frame this renderer can allocate.
        setTiles(width, height) {
            const perTile = 120000;
            const tiles = Math.min(16, Math.max(1, Math.ceil(Math.sqrt((width * height) / perTile))));

            tracer.tiles.set(tiles, tiles);

            return tiles * tiles;
        },

        // one sample is spread over this many renderSample() calls
        tileCount() {
            return Math.max(1, tracer.tiles.x) * Math.max(1, tracer.tiles.y);
        },

        // the volume uniforms follow the material texture setScene uploads, on the same material
        setScene(scene, camera) {
            clearMergedGroups();
            tracer.setScene(scene, camera);
            syncVolume(scene);
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
                    () => {
                        syncVolume(scene);

                        return true;
                    },
                    (e) => {
                        workerProbe = Promise.resolve(null);

                        throw e;
                    },
                );
            });
        },

        // What lens the camera is wearing, in the caller's units, or null before there is a
        // camera. Only the camera knows: setupCamera closes the aperture behind the caller's back.
        depthOfField() {
            if (!tracer.camera) {
                return null;
            }

            return {
                bokehSize: tracer.camera.bokehSize / 1000.0,
                focusDistance: tracer.camera.focusDistance,
                apertureBlades: tracer.camera.apertureBlades,
            };
        },

        // The lens lives on the camera object because that is the only place upstream reads
        // one from: its uniform copies nothing off a camera that is not a PhysicalCamera. Written
        // fresh on every change rather than stored, because bokehSize is an accessor over fStop
        // derived from the current focal length - which a window resize changes, and a stored
        // value would then read back as a different aperture than the one that was set.
        // False means there was no camera to write on yet.
        setDepthOfField(size, distance, blades) {
            if (!tracer.camera) {
                return false;
            }

            // the parameter is the aperture's diameter in scene units; the shader wants
            // millimetres against world units taken for metres, hence the thousand
            tracer.camera.bokehSize = size * 1000.0;
            tracer.camera.focusDistance = distance;
            // an iris of one or two sides is not a shape; upstream reads anything below three
            // as a circle only by accident, so say it here
            tracer.camera.apertureBlades = blades >= 3 ? blades : 0;
            tracer.updateCamera();

            return true;
        },

        updateCamera() {
            epoch += 1;
            tracer.updateCamera();
        },

        updateEnvironment() {
            epoch += 1;
            tracer.updateEnvironment();
        },

        // exact pixel size for screenshots; renderScale only lands within +-1px of the target
        setFixedSize(width, height) {
            epoch += 1;
            tracer.synchronizeRenderSize = false;
            tracer._pathTracer.setSize(width, height);

            // A screenshot resolution costs three more float targets of that size, about 400 MB
            // at 4K. Worth skipping when the buffer only feeds a number nobody reads while a
            // render is running - and not worth skipping when the denoiser is guided by it, or
            // the screenshot comes out unfiltered while the viewport is not.
            // A screenshot keeps the halves when something consumes them - the filter is
            // guided by them, and switching them off here is what made a screenshot come out
            // unfiltered while the viewport was not. It costs three float targets at screenshot
            // resolution, which is the price of a filtered screenshot and cannot be avoided:
            // the guide has to exist at the resolution being filtered.
            if (variance !== null && !varianceRequired) {
                wantVariance = variance.isEnabled() || wantVariance;
                variance.setEnabled(false);
            }
        },

        releaseFixedSize() {
            epoch += 1;
            tracer.synchronizeRenderSize = true;

            if (variance !== null && wantVariance) {
                variance.setEnabled(true);
            }
        },

        // Upstream's `target` getter is a fixed slot, `_blendTargets[1]`, but renderTask
        // swaps only its own locals and never reorders that array, so sample k lands in
        // `_blendTargets[k % 2]` and slot 1 holds a mean only after an ODD count. Read at
        // the default budget of 64 it hands back the 63-sample mean: the last sample of
        // every even accumulation is traced, blended, then not looked at. Pick the slot by
        // parity instead. Ceil, not round: mid-sample the destination is the one being
        // blended into, which is what upstream shows on its odd samples too.
        targetTexture() {
            const inner = tracer._pathTracer;

            if (!inner || !inner._alpha || !inner._blendTargets) {
                const fallback = tracer.target;

                return fallback.texture || fallback;
            }

            const target = inner._blendTargets[Math.ceil(inner.samples) % 2];

            return target.texture || target;
        },

        updateMaterials() {
            epoch += 1;
            tracer.updateMaterials();
            refreshMaterialIndex();
            // a roughness or metalness edit on the volume lands here, not in a rebuild
            syncVolume(tracer.scene);
        },

        // The sample counter cannot move while a program is compiling - both update()
        // and renderSample() return early on this flag - so a caller counting fruitless
        // iterations has to be able to tell a compile from a stall.
        isCompiling() {
            return Boolean(tracer && tracer.isCompiling);
        },

        renderSample() {
            tracer.renderSample();

            const { samples } = tracer;

            // the single funnel both sample loops go through, so the only hook needed
            if (variance !== null) {
                variance.afterRenderSample(tracer._pathTracer, samples, epoch);
            }

            return { samples };
        },

        reset() {
            epoch += 1;
            tracer.reset();

            if (variance !== null) {
                variance.reset();
            }
        },

        // Off by default and deliberately not a plot parameter yet: until something
        // consumes the number, a trait would be one more thing to keep registered in
        // five places. Reached through the diagnostic handle instead.
        setVariance(enabled, required = false) {
            wantVariance = Boolean(enabled);
            varianceRequired = wantVariance && Boolean(required);

            if (variance !== null) {
                variance.setEnabled(wantVariance);
            }
        },

        varianceHalves() {
            return variance === null ? null : variance.halves();
        },

        releaseBVHWorker,

        dispose() {
            releaseBVHWorker();

            if (variance !== null) {
                variance.dispose();
                variance = null;
            }

            if (tracer) {
                // drop the volume references and the diagnostic handle with the tracer
                const material = tracer._pathTracer && tracer._pathTracer.material;

                if (material && typeof material.clearVolume === 'function') {
                    material.clearVolume();
                }

                if (typeof window !== 'undefined' && window.__k3dTracer === tracer) {
                    delete window.__k3dTracer;
                }

                tracer.dispose();

                // the tracer disposes its own quad, not the material we installed on it, and
                // that material owns the majorant grid
                if (material && typeof material.dispose === 'function') {
                    material.dispose();
                }

                tracer = null;
            }
        },
    };
};
