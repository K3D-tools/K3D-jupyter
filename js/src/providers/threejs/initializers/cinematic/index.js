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
    let ready = false;
    let scene = null;
    let sceneDirty = true;
    let materialsDirty = false;
    let envKey = null;
    let lastBounces = null;
    let lastGlossyFilter = null;
    // the stratified-sample texture rebuilds on the first sample after a scene or bounce
    // change and consumes seeded RNG draws reset() does not replay: warm up one sample first
    let needsWarmup = true;
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

    // edits the tracer refreshes without a BVH rebuild. Colour stays out: for points and
    // tubes it is baked into vertex colours, so changing it is a geometry change.
    const MATERIAL_ONLY = ['roughness', 'metalness', 'opacity'];

    ['OBJECT_LOADED', 'OBJECT_REMOVED', 'OBJECT_CHANGE'].forEach((name) => {
        K3D.on(K3D.events[name], (change) => {
            if (change && MATERIAL_ONLY.indexOf(change.key) !== -1) {
                materialsDirty = true;
            } else {
                sceneDirty = true;
            }

            generation++;
        });
    });

    function restart() {
        generation++;
        needsWarmup = true;
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
        proxy.populate(scene, K3D.getWorld().camera);
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

    function startBuild(key) {
        const camera = K3D.getWorld().camera;
        const built = buildScene();
        const triangles = triangleCount(built);

        sceneDirty = false;
        materialsDirty = false;
        envKey = key;
        setHud('cinematic: building BVH…');

        if (triangles < workerTriangles) {
            backend.setScene(built, camera);
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

            lastBuild = { triangles, worker: offThread };
            needsWarmup = true;
        }, (e) => {
            // the generator holds the failed build and refuses a synchronous one: drop the
            // tracer so the retry starts from a clean one
            backend.dispose();
            ready = false;
            lastBounces = null;
            lastGlossyFilter = null;
            sceneDirty = true;
            onError(e);
        });
    }

    // Returns null when the tracer is traceable right now, or a promise resolving when an
    // off-thread build lands. Null rather than a resolved promise on purpose: a scene built on
    // the main thread has to stay in the caller's tick, or the frame that follows composites
    // one microtask later than the build and lands on a different image.
    function ensurePrepared() {
        if (!ready) {
            backend.init();
            ready = true;
        }

        // bounces alter only the tracer material - never the BVH
        if (lastBounces !== K3D.parameters.cinematicBounces) {
            backend.setBounces(K3D.parameters.cinematicBounces);
            lastBounces = K3D.parameters.cinematicBounces;
            needsWarmup = true;
        }

        const glossyFilter = K3D.parameters.cinematicGlossyFilter || 0.0;

        if (lastGlossyFilter !== glossyFilter) {
            backend.setGlossyFilter(glossyFilter);
            lastGlossyFilter = glossyFilter;
            needsWarmup = true;
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

    // headless yields through the task queue: rAF is throttled on a hidden page and would
    // stall the suite. The interactive path drives itself off rAF instead (see wake()).
    const yieldToBrowser = (fn) => setTimeout(fn, 0);

    function renderUntil(target, gen, budget, present, interruptible = true) {
        return new Promise((resolve, reject) => {
            const started = performance.now();
            // the counter does not advance while shaders compile or the library is paused;
            // without a ceiling on fruitless iterations the caller's promise never settles
            let idle = 0;
            let lastSamples = -1;

            function step() {
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
                    setHud(`cinematic: ${Math.floor(Math.min(samples, budget))} / ${budget} samples`);
                }

                if (samples >= target) {
                    // no gl.finish(): a fence stalls the CPU on the GPU drain, and the
                    // screenshot path's readback synchronises anyway
                    resolve({ samples, ms: performance.now() - started });
                    return;
                }

                if (samples === lastSamples) {
                    idle++;

                    // generous: shader compilation on a software renderer legitimately
                    // costs hundreds of idle iterations
                    if (idle > 5000) {
                        reject(new Error(`the path tracer stopped advancing at ${samples} `
                            + `of ${target} samples`));
                        return;
                    }
                } else {
                    idle = 0;
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

                // headless never looks at the canvas - the screenshot composes its own frame -
                // and presenting between samples costs a clear and two scene draws each time
                return renderSamplesAsync(budget, budget, !isHeadless);
            })).then((result) => {
                if (!result.stale) {
                    setHud(`cinematic: ${result.samples} / ${budget} samples`);
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

                        // a moving camera discards the accumulation every frame, leaving
                        // nothing traced to show: rasterise this frame instead
                        if (rasterizePreview !== null) {
                            rasterizePreview();
                            setHud(`cinematic: 0 / ${budget} samples`);
                            frameHandle = window.requestAnimationFrame(frame);

                            return;
                        }
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
                setHud(`cinematic: ${Math.floor(Math.min(samples, budget))} / ${budget} samples`);

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
