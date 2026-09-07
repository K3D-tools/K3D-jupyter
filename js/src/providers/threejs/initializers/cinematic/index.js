// Cinematic mode orchestration: a proxy scene of plain meshes, the environment as the
// sole light, an interruptible accumulation loop.
const THREE = require('three');
const createWebGLBackend = require('./webglBackend');
const createSceneProxy = require('./sceneProxy');
const { getEnvironmentTexture } = require('../../helpers/environment');

// must match Scene.applyRendererMode: both renderer modes orient the environment alike
function environmentRotation(K3D) {
    const rot = K3D.parameters.environmentRotation;

    switch (K3D.parameters.cameraUpAxis) {
        case 'y':
            return new THREE.Euler(0, rot, 0, 'XYZ');
        case 'x':
            return new THREE.Euler(rot, 0, -Math.PI / 2, 'XZY');
        default:
            return new THREE.Euler(Math.PI / 2, 0, rot, 'ZXY');
    }
}

// Where the sharp plane sits, resolved the one way: the parameter when it is set, and
// otherwise whatever the camera is pointed at. The focus helper draws this same number, so it
// has to come from here too, or the green plane would lie about where the focus is.
function resolveFocusDistance(K3D) {
    const explicit = K3D.parameters.cinematicFocusDistance || 0.0;

    if (explicit > 0.0) {
        return explicit;
    }

    const world = K3D.getWorld();

    return world.controls && world.controls.target
        ? world.camera.position.distanceTo(world.controls.target)
        : world.camera.position.length();
}

module.exports = function cinematic(K3D, renderer, hooks) {
    const backend = createWebGLBackend(renderer);
    const proxy = createSceneProxy(K3D);
    const presentFrame = (hooks && hooks.presentFrame) || null;
    const prepareOverlay = (hooks && hooks.prepareOverlay) || null;
    const onError = (hooks && hooks.onError) || ((e) => { throw e; });
    const rasterizePreview = (hooks && hooks.rasterizePreview) || null;
    const isHeadless = typeof window !== 'undefined'
        && typeof window.headlessK3D !== 'undefined';
    // wanted: the image is not converged yet
    let wanted = false;
    let frameHandle = null;
    // not every camera path emits CAMERA_CHANGE, so the loop compares matrices
    const lastCamera = { view: new THREE.Matrix4(), projection: new THREE.Matrix4() };
    let cameraKnown = false;
    let scene = null;
    let sceneDirty = true;
    let materialsDirty = false;
    let envKey = null;
    let lastBounces = null;
    let lastGlossyFilter = null;
    let lastDenoise = 0.0;
    // undefined, not null: null is a legal seed, and the first pass must still apply it
    let lastSeed;
    // the stratified-sample texture rebuilds on the first sample after a scene or bounce
    // change and consumes seeded RNG draws reset() does not replay: warm up one sample first
    let needsWarmup = true;
    // the first draw of a fresh program pays for compiling it - 16 s for the volume shader on a
    // laptop 4070 through ANGLE - and the call blocks, so the notice has to be painted first
    let announceCompile = true;
    // a bump strands every in-flight accumulation loop
    let generation = 0;
    let hud = null;
    let hudText = null;
    // Above this the BVH goes to a worker: the build is one uninterruptible block, and on a
    // few million triangles it stops answering the OS, taking the whole browser down rather
    // than one tab. Below it, the build is shorter than the worker round trip.
    const WORKER_TRIANGLES = 100000;
    let workerTriangles = WORKER_TRIANGLES;
    let buildPromise = null;
    let lastBuild = { triangles: 0, worker: false };
    // The camera stopping for a single frame is not the end of an interaction: a one-sample
    // trace presented over the rasterised preview reads as flicker while orbiting. The library's
    // own loop waits out renderDelay for the same reason, and cross-fades on top of it.
    const SETTLE_MS = 200;
    let settleUntil = 0;

    // Spend no samples until the interaction stops: a slider sends a change per pixel of the
    // drag and each one abandons the accumulation, so tracing between them only ever presents a
    // one-sample frame.
    function holdSamples() {
        settleUntil = performance.now() + SETTLE_MS;
    }

    // edits the tracer refreshes without a BVH rebuild, whatever the object is
    const MATERIAL_ONLY = ['roughness', 'metalness', 'opacity', 'light_scale'];
    // and these on a volume only - on points and tubes the same keys are baked into vertex
    // attributes, where changing one is geometry
    const VOLUME_MATERIAL_ONLY = ['color_range', 'alpha_coef', 'gradient_step'];

    // OBJECT_CHANGE names one key, OBJECT_LOADED a whole set; no payload means assume the worst
    function materialOnly(change) {
        if (!change || typeof change !== 'object') {
            return false;
        }

        const keys = change.keys || (typeof change.key === 'string' ? [change.key] : null);

        if (keys === null || keys.length === 0) {
            return false;
        }

        const json = K3D.getWorld().ObjectsListJson[change.id];
        const volume = Boolean(json) && json.type === 'Volume';

        return keys.every((key) => MATERIAL_ONLY.indexOf(key) !== -1
            || (volume && VOLUME_MATERIAL_ONLY.indexOf(key) !== -1));
    }

    ['OBJECT_LOADED', 'OBJECT_REMOVED', 'OBJECT_CHANGE'].forEach((name) => {
        K3D.on(K3D.events[name], (change) => {
            if (materialOnly(change)) {
                materialsDirty = true;
            } else {
                if (name === 'OBJECT_LOADED') {
                    // the event does not always name an object, so only a full drop is safe
                    proxy.invalidate();
                }

                sceneDirty = true;
            }

            holdSamples();
            generation++;
        });
    });

    function restart() {
        generation++;
        needsWarmup = true;
        holdSamples();
        backend.reset();
    }

    // latches the compared matrices as a side effect
    function cameraMoved() {
        const { camera } = K3D.getWorld();

        camera.updateMatrixWorld();

        if (cameraKnown
            && lastCamera.view.equals(camera.matrixWorld)
            && lastCamera.projection.equals(camera.projectionMatrix)) {
            return false;
        }

        lastCamera.view.copy(camera.matrixWorld);
        lastCamera.projection.copy(camera.projectionMatrix);
        cameraKnown = true;

        return true;
    }

    // the loop detects the move itself; this only wakes it up when it had parked on a
    // converged image. Headless is excluded on purpose: there frames are rendered on request.
    K3D.on(K3D.events.CAMERA_CHANGE, () => {
        if (K3D.parameters.renderer === 'cinematic' && !isHeadless) {
            K3D.render();
        }
    });

    function currentEnvKey() {
        const env = K3D.parameters.environment;

        return [
            typeof env === 'string' ? env : (env && env.name) || 'custom',
            K3D.parameters.environmentRotation,
            K3D.parameters.cameraUpAxis,
            K3D.parameters.lighting,
        ].join('|');
    }

    function applyEnvironment(target) {
        const env = getEnvironmentTexture(K3D.parameters.environment);
        const rotation = environmentRotation(K3D);
        // the curve Scene.recalculateLights uses, so plot.lighting means the same here
        const lighting = K3D.parameters.lighting;
        const envIntensity = lighting <= 1.0 ? Math.max(lighting, 0.0) : (1.0 + lighting) / 2.0;

        env.mapping = THREE.EquirectangularReflectionMapping;
        target.environment = env;
        target.environmentRotation.copy(rotation);
        // advanced's measured 1.2 surface correction, with no cinematic gain on top:
        // isolated flat surfaces are legitimately darker, lit only by what reaches them.
        target.environmentIntensity = envIntensity * 1.2;

        // not the backdrop: the frame stays transparent where nothing was hit, so
        // background_color shows through. A THREE.Color here renders black.
        target.background = null;
    }

    function buildScene() {
        scene = new THREE.Scene();
        proxy.populate(scene, K3D.getWorld().camera, { volumes: backend.volumeSupported() });
        applyEnvironment(scene);

        return scene;
    }

    function triangleCount(target) {
        let triangles = 0;

        target.traverse((node) => {
            const { geometry } = node;

            if (geometry && geometry.attributes.position) {
                triangles += (geometry.index
                    ? geometry.index.count
                    : geometry.attributes.position.count) / 3;
            }
        });

        return triangles;
    }

    function ensureHud() {
        if (hud !== null) {
            return hud;
        }

        hud = document.createElement('div');
        hud.style.cssText = 'position:absolute;top:0;left:0;padding:3px 6px;'
            + 'background:rgba(0,0,0,0.65);color:#eee;font:12px monospace;'
            + 'z-index:10;pointer-events:none;display:none;';
        K3D.getWorld().targetDOMNode.appendChild(hud);

        return hud;
    }

    // The counter says how much has been spent; this says whether it was enough. Silent
    // until the variance buffer is on and its first readback has landed, so a plot nobody
    // asked to measure reads exactly as it did before.
    function sampleHud(samples, budget) {
        // A fresh program costs about fifteen seconds on a volume scene, and the sample counter
        // cannot move until it links - so without this the renderer looks stopped at zero for as
        // long as the driver takes. Ask the tracer rather than latching a flag, because every new
        // variant compiles again: opening the aperture is a different program from closing it.
        if (backend.isReady() && backend.isCompiling()) {
            return 'cinematic: compiling shader…';
        }

        const whole = Math.floor(Math.min(samples, budget));

        return `cinematic: ${whole} / ${budget} samples`;
    }

    function setHud(text) {
        const node = ensureHud();

        if (text === null) {
            node.style.display = 'none';
            hudText = null;

            return;
        }

        // the counter is touched once per tile; rewriting the same string flickers
        if (text !== hudText) {
            node.textContent = text;
            hudText = text;
        }

        node.style.display = 'block';
    }

    // ensurePrepared runs before the scene reaches the tracer, so on a first prepare there is
    // no camera to put a lens on - which is why the scene handover calls this again.
    function applyDepthOfField() {
        // a pinhole is the default, and then the focus distance cannot matter - resolving it
        // anyway would recompute and reset on every camera move for an effect that is off
        const bokehSize = K3D.parameters.cinematicBokehSize || 0.0;
        let focusDistance = 0.0;

        if (bokehSize > 0.0) {
            focusDistance = resolveFocusDistance(K3D);
        }

        const apertureBlades = bokehSize > 0.0
            ? (K3D.parameters.cinematicApertureBlades || 0) : 0;
        const current = backend.depthOfField();

        // against the camera, never against what we remember applying: setupCamera closes the
        // aperture, and Core calls it on every assignment to plot.camera - once a frame in an
        // animation, which is where a remembered value silently loses the lens for good
        if (current !== null
            && Math.abs(current.bokehSize - bokehSize) < 1e-9
            && current.focusDistance === focusDistance
            && current.apertureBlades === (apertureBlades >= 3 ? apertureBlades : 0)) {
            return false;
        }

        return backend.setDepthOfField(bokehSize, focusDistance, apertureBlades);
    }

    function startBuild(key) {
        const camera = K3D.getWorld().camera;
        const built = buildScene();
        const triangles = triangleCount(built);

        sceneDirty = false;
        materialsDirty = false;
        envKey = key;
        setHud('cinematic: building BVH…');
        announceCompile = true;

        if (triangles < workerTriangles) {
            backend.setScene(built, camera);
            applyDepthOfField();
            lastBuild = { triangles, worker: false };
            needsWarmup = true;

            return null;
        }

        return backend.setSceneAsync(built, camera, (progress) => {
            setHud(`cinematic: building BVH… ${Math.round(progress * 100)}%`);
        }).then((offThread) => {
            if (!offThread) {
                backend.setScene(built, camera);
            }

            applyDepthOfField();
            lastBuild = { triangles, worker: offThread };
            needsWarmup = true;
        }, (e) => {
            // the generator holds the failed build and refuses a synchronous one: drop the
            // tracer so the retry starts from a clean one
            backend.dispose();
            lastBounces = null;
            lastGlossyFilter = null;
            lastSeed = undefined;
            sceneDirty = true;
            onError(e);
        });
    }

    // Returns null when the tracer is traceable right now, or a promise resolving when an
    // off-thread build lands. Null rather than a resolved promise on purpose: a scene built on
    // the main thread has to stay in the caller's tick, or the frame that follows composites
    // one microtask later than the build and lands on a different image.
    function ensurePrepared() {
        if (!backend.isReady()) {
            backend.init();
        }

        // bounces alter only the tracer material - never the BVH
        if (lastBounces !== K3D.parameters.cinematicBounces) {
            backend.setBounces(K3D.parameters.cinematicBounces);
            lastBounces = K3D.parameters.cinematicBounces;
            needsWarmup = true;
            holdSamples();
        }

        // The filter is guided by the two halves of the accumulation, which fill one blend
        // per completed sample. Switching it on part way through finds them empty and there is
        // no way to fill them for samples already spent, so this restarts - which is also what
        // makes switching it off show the raw image straight away.
        const denoise = K3D.parameters.cinematicDenoise || 0.0;

        if (lastDenoise !== denoise) {
            const wasOn = lastDenoise > 0.0;
            const isOn = denoise > 0.0;

            lastDenoise = denoise;

            if (wasOn !== isOn) {
                // required: a fixed-size render must not switch the buffer off underneath the
                // filter, or a screenshot comes out unfiltered while the viewport is not
                backend.setVariance(isOn, true);
                needsWarmup = true;
                holdSamples();
            }
        }

        const glossyFilter = K3D.parameters.cinematicGlossyFilter || 0.0;

        if (lastGlossyFilter !== glossyFilter) {
            backend.setGlossyFilter(glossyFilter);
            lastGlossyFilter = glossyFilter;
            needsWarmup = true;
            holdSamples();
        }

        if (applyDepthOfField()) {
            needsWarmup = true;
            holdSamples();
        }

        const seed = K3D.parameters.cinematicSeed === undefined ? null : K3D.parameters.cinematicSeed;

        if (lastSeed !== seed) {
            backend.setSeed(seed);
            lastSeed = seed;
            needsWarmup = true;
            holdSamples();
        }

        const key = currentEnvKey();

        // A build in flight makes the tracer unusable for everyone, not just the caller that
        // started it: startBuild clears the dirty flags when it dispatches, so a second caller
        // would otherwise read the scene as ready and accumulate against the old one.
        if (buildPromise !== null) {
            return buildPromise.then(ensurePrepared);
        }

        if (sceneDirty) {
            if (buildPromise === null) {
                const building = startBuild(key);

                if (building === null) {
                    return null;
                }

                buildPromise = building.then(() => {
                    buildPromise = null;
                });
            }

            // re-entered after the build: a scene change that arrived meanwhile is picked up
            // here, and a failed off-thread build retries on the main thread
            return buildPromise.then(ensurePrepared);
        }

        if (materialsDirty) {
            // BVH untouched: refresh the material texture only
            proxy.syncMaterials();
            backend.updateMaterials();
            materialsDirty = false;
        }

        if (key !== envKey) {
            // lighting-only change: no BVH rebuild
            applyEnvironment(scene);
            backend.updateEnvironment();
            envKey = key;
            holdSamples();
        }

        return null;
    }

    function buildInFlight() {
        return buildPromise !== null;
    }

    // the tracer is usually ready in this very tick; only an off-thread build defers the work
    function prepared(run) {
        const building = ensurePrepared();

        return building === null ? run() : building.then(run);
    }

    // wall clock, not a turn count: the yield below is not throttled, so turns say nothing
    const STALL_MS = 20000;

    // Headless yields through the task queue: rAF is throttled on a hidden page and would
    // stall the suite. The interactive path drives itself off rAF instead (see wake()).
    // Not setTimeout: Chrome clamps a self-rescheduling chain to ~4.4 ms, and this one turns
    // once per tile - at 4K that alone floors a 128-sample frame at 45 s.
    const yieldToBrowser = (() => {
        if (typeof MessageChannel === 'undefined') {
            return (fn) => setTimeout(fn, 0);
        }

        const channel = new MessageChannel();
        const pending = [];

        channel.port1.onmessage = () => {
            const fn = pending.shift();

            if (fn) {
                fn();
            }
        };

        return (fn) => {
            pending.push(fn);
            channel.port2.postMessage(0);
        };
    })();

    function renderUntil(target, gen, budget, present, interruptible = true) {
        return new Promise((resolve, reject) => {
            const started = performance.now();
            // the counter does not advance while shaders compile or the library is paused;
            // without a ceiling on fruitless waiting the caller's promise never settles
            let idleSince = null;
            let lastSamples = -1;

            function step() {
                if (announceCompile) {
                    setHud('cinematic: compiling shader…');
                    announceCompile = false;
                }

                if (interruptible && gen !== generation) {
                    resolve({ samples: 0, ms: 0, stale: true });
                    return;
                }

                if (K3D.disabling) {
                    resolve({ samples: 0, ms: 0, stale: true });
                    return;
                }

                let samples;

                try {
                    // a throw from a setTimeout continuation escapes the promise otherwise
                    samples = backend.renderSample().samples;
                } catch (e) {
                    reject(e);
                    return;
                }

                if (present && presentFrame !== null) {
                    presentFrame(backend.targetTexture());
                }

                if (budget) {
                    // whole samples: the counter advances by a fraction per tile
                    setHud(sampleHud(samples, budget));
                }

                if (samples >= target) {
                    // no gl.finish(): a fence stalls the CPU on the GPU drain, and the
                    // screenshot path's readback synchronises anyway
                    resolve({ samples, ms: performance.now() - started });
                    return;
                }

                if (samples === lastSamples) {
                    // the clock runs only while nothing is compiling, or this would cap
                    // compile time rather than catch a stall - the volume shader takes ~16 s
                    if (backend.isCompiling()) {
                        idleSince = null;
                    } else if (idleSince === null) {
                        idleSince = performance.now();
                    } else if (performance.now() - idleSince > STALL_MS) {
                        reject(new Error(`the path tracer stopped advancing at ${samples} `
                            + `of ${target} samples`));
                        return;
                    }
                } else {
                    idleSince = null;
                    lastSamples = samples;
                }

                yieldToBrowser(step);
            }

            step();
        });
    }

    // async because samples are skipped while shaders compile in the background
    // (KHR_parallel_shader_compile): yield until the accumulator reaches the budget
    function renderSamplesAsync(count, budget, present, interruptible = true) {
        const gen = ++generation;
        const warmup = needsWarmup
            ? renderUntil(1, gen, 0, false, interruptible)
            : Promise.resolve({ samples: 0 });

        // rewind to the canonical RNG state before every accumulation
        return warmup.then((first) => {
            if (first.stale) {
                return first;
            }

            needsWarmup = false;
            backend.reset();

            return renderUntil(count, gen, budget, present, interruptible);
        });
    }

    return {
        isSupported: backend.isSupported,
        unsupportedReason: backend.unsupportedReason,

        prepare() {
            return Promise.resolve(prepared(() => undefined));
        },

        // accumulate to the plot's sample budget, presenting every sample on the canvas;
        // resolves with stale: true when superseded
        renderFrame() {
            // nobody can see a headless canvas, and a screenshot traces its own frame from zero
            if (isHeadless) {
                return Promise.resolve({ samples: 0, ms: 0 });
            }

            const budget = K3D.parameters.cinematicSamples;
            const world = K3D.getWorld();

            // the prologue can throw (proxy, BVH, volume layer); it runs inside the chain so
            // the caller's .catch sees it
            return Promise.resolve().then(() => prepared(() => {
                backend.updateCamera();
                backend.setTiles(world.width, world.height);

                if (prepareOverlay !== null) {
                    prepareOverlay(scene, world.width, world.height);
                }

                return renderSamplesAsync(budget, budget, true);
            })).then((result) => {
                if (!result.stale) {
                    setHud(sampleHud(result.samples, budget));
                }

                return result;
            });
        },

        // offscreen accumulation at an explicit resolution (screenshots); resolves with the
        // float target texture, which the caller blits through the shared tone-mapping pass
        renderBudget(width, height) {
            const budget = K3D.parameters.cinematicSamples;

            return Promise.resolve().then(() => prepared(() => {
                backend.updateCamera();
                backend.setFixedSize(width, height);
                backend.setTiles(width, height);

                if (prepareOverlay !== null) {
                    prepareOverlay(scene, width, height);
                }

                // no canvas presentation: the offscreen target has its own size, so a blit
                // would corrupt the preview. Uninterruptible on purpose - the sync that
                // precedes a screenshot fires the scene events that abandon accumulations.
                return renderSamplesAsync(budget, budget, false, false);
            })).then((result) => ({
                samples: result.samples,
                texture: backend.targetTexture(),
            }));
        },

        releaseFixedSize() {
            backend.releaseFixedSize();
            backend.reset();
        },

        // one renderSample() per animation frame, never a loop inside a single task: the
        // browser composites between frames and an edit lands on the next one.
        wake() {
            wanted = true;

            if (frameHandle !== null) {
                return;
            }

            const frame = () => {
                frameHandle = null;

                if (K3D.parameters.renderer !== 'cinematic' || K3D.disabling || !wanted) {
                    return;
                }

                // a detached target node is the only reliable signal that this instance is
                // gone: a replaced widget need not call disable(), and the rAF loop outlives it
                const node = K3D.getWorld().targetDOMNode;

                if (!node || node.isConnected === false) {
                    wanted = false;
                    setHud(null);
                    backend.releaseBVHWorker();

                    return;
                }

                const budget = K3D.parameters.cinematicSamples;
                const world = K3D.getWorld();
                let samples;

                try {
                    const building = ensurePrepared();

                    if (building !== null) {
                        // the next frame calls in synchronously and reports what went wrong
                        building.catch(() => {});
                    }

                    // an off-thread BVH build: nothing to trace against yet, so keep the
                    // camera live on rasterised frames instead of blocking on it
                    if (buildInFlight()) {
                        if (rasterizePreview !== null) {
                            rasterizePreview();
                        }

                        frameHandle = window.requestAnimationFrame(frame);

                        return;
                    }

                    backend.setTiles(world.width, world.height);

                    if (cameraMoved()) {
                        backend.updateCamera();
                        restart();
                    }

                    // a moving camera discards the accumulation every frame, leaving nothing
                    // traced to show, and a camera that has just stopped is probably still being
                    // dragged: rasterise until it holds still
                    // the focus helper is a rasterised affordance: while it is up there is
                    // nothing to accumulate, and tracing behind it would only waste the GPU
                    const helper = K3D.getWorld().focusHelper === true;

                    if ((helper || performance.now() < settleUntil) && rasterizePreview !== null) {
                        rasterizePreview();
                        setHud(helper
                            ? 'cinematic: focus helper - tracing paused'
                            : sampleHud(0, budget));
                        frameHandle = window.requestAnimationFrame(frame);

                        return;
                    }

                    if (announceCompile) {
                        // the browser needs a frame to paint it before the blocking call
                        setHud('cinematic: compiling shader…');
                        announceCompile = false;
                        frameHandle = window.requestAnimationFrame(frame);

                        return;
                    }

                    if (needsWarmup) {
                        // rewind after the sample texture rebuild, so an interactive frame
                        // converges to the same image a screenshot does
                        backend.renderSample();
                        backend.reset();
                        needsWarmup = false;
                    }

                    samples = backend.renderSample().samples;

                    if (presentFrame !== null) {
                        presentFrame(backend.targetTexture());
                    }
                } catch (e) {
                    wanted = false;
                    setHud(null);
                    onError(e);

                    return;
                }

                // samples advance by a fraction per tile - report whole ones
                setHud(sampleHud(samples, budget));

                if (samples >= budget) {
                    wanted = false;
                    K3D.dispatch(K3D.events.RENDERED);

                    return;
                }

                frameHandle = window.requestAnimationFrame(frame);
            };

            frameHandle = window.requestAnimationFrame(frame);
        },

        // building blocks for headless probes and benchmarks
        renderSample() {
            return backend.renderSample();
        },

        renderSamplesAsync(count) {
            // probes hash the canvas, so the accumulation must reach it
            return renderSamplesAsync(count, count, true);
        },

        // scene size decides where the BVH is built, so a probe needs to move the threshold
        // to drive both paths over one scene; null puts it back
        setWorkerThreshold(triangles) {
            workerTriangles = (triangles === null) ? WORKER_TRIANGLES : triangles;
        },

        lastBuild() {
            return lastBuild;
        },

        // How far from converged the accumulation is, as a relative RMS error of the
        // mean. Off by default and reached only from here: it costs three float
        // targets and nothing consumes the number yet, so it is not a plot parameter.
        setVariance(enabled, required) {
            backend.setVariance(enabled, required);
        },

        // Re-compose what is already accumulated, without tracing another sample. A finished
        // render presents nothing more on its own, so anything that changes how the accumulation
        // is turned into pixels - the denoiser above all - would otherwise not show until the
        // next sample, which may never come.
        present() {
            if (presentFrame === null || !backend.isReady()) {
                return false;
            }

            presentFrame(backend.targetTexture());

            return true;
        },

        varianceHalves() {
            return backend.varianceHalves();
        },

        hideHud() {
            if (hud !== null) {
                hud.style.display = 'none';
            }
        },

        // strands any in-flight accumulation without discarding prepared state
        abort() {
            generation++;
            wanted = false;

            if (frameHandle !== null) {
                window.cancelAnimationFrame(frameHandle);
                frameHandle = null;
            }
        },

        // back to sample zero with the new state, stranding in-flight loops
        restart() {
            restart();
        },

        invalidateScene() {
            proxy.invalidate();
            sceneDirty = true;
        },

        updateCamera: backend.updateCamera,
        reset: backend.reset,
        dispose: backend.dispose,
    };
};

module.exports.resolveFocusDistance = resolveFocusDistance;
