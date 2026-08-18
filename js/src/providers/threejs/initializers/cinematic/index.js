// Cinematic mode orchestration: the scene proxy mirrors K3DObjects as plain
// meshes (stage 2), the environment is the sole light, the accumulation loop
// is interruptible and a HUD counts samples. Screenshots and tone mapping
// integration are stage 3.
const THREE = require('three');
const createWebGLBackend = require('./webglBackend');
const createSceneProxy = require('./sceneProxy');
const { getEnvironmentTexture } = require('../../helpers/environment');

// the same up-axis-dependent Euler the advanced renderer applies in
// Scene.applyRendererMode - the two modes must agree on where the sun sits
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
    // interactive loop state: `wanted` is "the image is not converged yet",
    // frameHandle is the pending animation frame
    let wanted = false;
    let frameHandle = null;
    // the camera reaches us through several paths (controls, the kernel setting
    // plot.camera, resetCamera, viewport changes) and only one of them emits
    // CAMERA_CHANGE - so the loop compares the matrices instead of trusting an
    // event, and a moved camera always restarts the accumulation
    const lastCamera = { view: new THREE.Matrix4(), projection: new THREE.Matrix4() };
    let cameraKnown = false;
    let ready = false;
    let scene = null;
    let sceneDirty = true;
    let envKey = null;
    let lastBounces = null;
    // the stratified-sample texture rebuilds on the first sample after a scene
    // or bounce change, consuming seeded RNG draws that a later reset() does
    // not replay - those accumulations warm up one sample first (stage 0 gate)
    let needsWarmup = true;
    // bumping the generation strands every in-flight accumulation loop, so a
    // camera move or scene edit never waits for a full sample budget
    let generation = 0;
    let hud = null;
    let hudText = null;

    // the source of truth is K3DObjects - any object mutation invalidates the
    // proxy scene (BVH rebuild), which ensurePrepared() picks up lazily. The
    // generation bump abandons whatever is accumulating right now: it is an
    // image of a scene that no longer exists, so finishing its budget would
    // only delay the one the user asked for.
    ['OBJECT_LOADED', 'OBJECT_REMOVED', 'OBJECT_CHANGE'].forEach((name) => {
        K3D.on(K3D.events[name], () => {
            sceneDirty = true;
            generation++;
        });
    });

    function restart() {
        // the accumulated image describes a state that no longer holds
        generation++;
        needsWarmup = true;
        backend.reset();
    }

    // true when the camera moved since the last frame this loop rendered
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

    // a moved camera invalidates the accumulation completely, so it restarts
    // from zero rather than refining a stale image. Headless is the exception:
    // there every frame is explicitly requested (sync, screenshot), and a
    // progressive preview nobody looks at would double the cost of every
    // reference - a camera reset would accumulate at canvas resolution just
    // before the screenshot accumulates at its own.
    // the loop detects the move itself; this only wakes it up when it had
    // already parked on a converged image
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
        // the exposure curve advanced uses (Scene.recalculateLights), so that
        // plot.lighting means the same thing in both PBR renderers
        const lighting = K3D.parameters.lighting;
        const envIntensity = lighting <= 1.0 ? Math.max(lighting, 0.0) : (1.0 + lighting) / 2.0;

        env.mapping = THREE.EquirectangularReflectionMapping;
        target.environment = env;
        target.background = env;
        target.environmentRotation.copy(rotation);
        target.backgroundRotation.copy(rotation);
        // Measured, not guessed: a watertight white cube in a uniform
        // environment leaves 2.45x less radiance here than under the raster IBL
        // of advanced, and the factor is constant across albedo (0.05..1.0), so
        // it is exposure rather than shading. Advanced carries the same kind of
        // measured correction for its own delivery (the 1.2 in Scene.js). The
        // gain applies to the background too - a metal must reflect the same
        // radiance that lights it - which also lands the backdrop near the white
        // one advanced draws.
        target.environmentIntensity = envIntensity * 1.2 * 1.633;
        target.backgroundIntensity = envIntensity * 1.2 * 1.633;
    }

    function buildScene() {
        scene = new THREE.Scene();
        proxy.populate(scene, K3D.getWorld().camera);
        applyEnvironment(scene);

        return scene;
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

        // the counter is touched several times per sample (once per tile), and
        // rewriting the same string makes the readout flicker
        if (text !== hudText) {
            node.textContent = text;
            hudText = text;
        }

        node.style.display = 'block';
    }

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

        const key = currentEnvKey();

        if (sceneDirty) {
            setHud('cinematic: building BVH…');
            backend.setScene(buildScene(), K3D.getWorld().camera);
            sceneDirty = false;
            envKey = key;
            needsWarmup = true;
        } else if (key !== envKey) {
            // lighting-only change: the BVH is untouched, no rebuild
            applyEnvironment(scene);
            backend.updateEnvironment();
            envKey = key;
        }
    }

    // Headless yields through the task queue: there is no compositor to sync
    // with, and rAF is throttled when the page is not visible, which would
    // stall the suite. The interactive path does not use this - it runs the
    // library's own animate() rhythm, one renderSample per frame (see wake()).
    const yieldToBrowser = (fn) => setTimeout(fn, 0);

    function renderUntil(target, gen, budget, present) {
        return new Promise((resolve, reject) => {
            const started = performance.now();
            // the library refuses to advance the counter while its shaders
            // compile or when it considers itself paused; without a ceiling on
            // fruitless iterations a stuck tracer spins this chain forever and
            // the caller's promise never settles
            let idle = 0;
            let lastSamples = -1;

            function step() {
                if (gen !== generation) {
                    resolve({ samples: 0, ms: 0, stale: true });
                    return;
                }

                if (K3D.disabling) {
                    resolve({ samples: 0, ms: 0, stale: true });
                    return;
                }

                let samples;

                try {
                    // a throw from a setTimeout continuation would otherwise
                    // escape the promise and leave the caller pending forever
                    samples = backend.renderSample().samples;
                } catch (e) {
                    reject(e);
                    return;
                }

                if (present && presentFrame !== null) {
                    presentFrame(backend.targetTexture());
                }

                if (budget) {
                    setHud(`cinematic: ${samples} / ${budget} samples`);
                }

                if (samples >= target) {
                    // no gl.finish() here: a fence stalls the CPU until the GPU
                    // drains, which is exactly the stutter this loop avoids.
                    // The readback in the screenshot path synchronises anyway.
                    resolve({ samples, ms: performance.now() - started });
                    return;
                }

                if (samples === lastSamples) {
                    idle++;

                    // generous: one sample spans a whole tile grid, and shader
                    // compilation legitimately costs hundreds of idle iterations
                    // on a software renderer
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

    // async by necessity: the library skips samples while its shaders compile
    // in the background (KHR_parallel_shader_compile), so the loop must yield
    // to the event loop until the accumulator reaches the budget
    function renderSamplesAsync(count, budget, present) {
        const gen = ++generation;
        const warmup = needsWarmup
            ? renderUntil(1, gen, 0, false)
            : Promise.resolve({ samples: 0 });

        // rewind to the canonical RNG state before every accumulation - after
        // the warm-up when the sample texture just rebuilt, directly otherwise
        return warmup.then((first) => {
            if (first.stale) {
                return first;
            }

            needsWarmup = false;
            backend.reset();

            return renderUntil(count, gen, budget, present);
        });
    }

    return {
        isSupported: backend.isSupported,
        unsupportedReason: backend.unsupportedReason,

        prepare() {
            ensurePrepared();
        },

        // one full progressive frame: rebuild what is dirty, then accumulate
        // up to the plot's sample budget; every sample lands on the canvas as
        // it converges. Resolves early (stale: true) when superseded.
        renderFrame() {
            const budget = K3D.parameters.cinematicSamples;
            const world = K3D.getWorld();

            // the prologue is synchronous but must fail like the loop does:
            // building the proxy, the BVH or the volume layer can throw, and
            // outside the chain that throw would escape the caller's .catch
            return Promise.resolve().then(() => {
                ensurePrepared();
                backend.updateCamera();
                backend.setTiles(world.width, world.height);

                if (prepareOverlay !== null) {
                    prepareOverlay(scene, world.width, world.height);
                }

                return renderSamplesAsync(budget, budget, true);
            }).then((result) => {
                if (!result.stale) {
                    setHud(`cinematic: ${result.samples} / ${budget} samples`);
                }

                return result;
            });
        },

        // offscreen accumulation at an explicit resolution (screenshots):
        // resolves with the float target texture holding the converged frame,
        // which the caller blits through the shared tone-mapping pass
        renderBudget(width, height) {
            const budget = K3D.parameters.cinematicSamples;

            return Promise.resolve().then(() => {
                ensurePrepared();
                backend.updateCamera();
                backend.setFixedSize(width, height);
                backend.setTiles(width, height);

                if (prepareOverlay !== null) {
                    prepareOverlay(scene, width, height);
                }

                // no canvas presentation: the offscreen target has its own
                // size, and every sample blitted to a differently sized canvas
                // would be both wasted work and a corrupted preview
                return renderSamplesAsync(budget, budget, false);
            }).then((result) => {
                if (result.stale) {
                    // a superseded accumulation holds no valid image; letting
                    // it through would hand a blank frame to a screenshot
                    throw new Error('the cinematic accumulation was superseded');
                }

                return {
                    samples: result.samples,
                    texture: backend.targetTexture(),
                };
            });
        },

        releaseFixedSize() {
            backend.releaseFixedSize();
            backend.reset();
        },

        // The interactive rhythm, straight out of the library's own animate():
        // one renderSample() per animation frame, forever, instead of a loop
        // inside a single task. The browser composites between frames (no
        // freeze, no driver watchdog), an edit takes effect on the very next
        // frame (no waiting out the remaining budget), and the loop parks itself
        // once the image has converged so an idle plot costs nothing.
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

                const budget = K3D.parameters.cinematicSamples;
                const world = K3D.getWorld();
                let samples;

                try {
                    ensurePrepared();
                    backend.setTiles(world.width, world.height);

                    if (cameraMoved()) {
                        backend.updateCamera();
                        restart();

                        // While the camera is being dragged, every frame throws
                        // the accumulation away, so path tracing has nothing to
                        // show yet. Rasterise the scene for this frame instead -
                        // the same trick the library's rasterizeScene does - so
                        // the plot follows the mouse instead of freezing on a
                        // stale image until the button is released.
                        if (rasterizePreview !== null) {
                            rasterizePreview();
                            setHud(`cinematic: 0 / ${budget} samples`);
                            frameHandle = window.requestAnimationFrame(frame);

                            return;
                        }
                    }

                    if (needsWarmup) {
                        // the sample texture rebuilds on the first sample after
                        // a scene change; rewind afterwards so an interactive
                        // frame converges to the same image a screenshot does
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

        hideHud() {
            if (hud !== null) {
                hud.style.display = 'none';
            }
        },

        // strands any in-flight accumulation without touching prepared state:
        // the headless loop sees the generation bump, the interactive loop stops
        // asking for frames
        abort() {
            generation++;
            wanted = false;

            if (frameHandle !== null) {
                window.cancelAnimationFrame(frameHandle);
                frameHandle = null;
            }
        },

        // an edit or a camera move: the accumulation starts over from sample
        // zero with the new state instead of refining the old image
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
