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

    // the source of truth is K3DObjects - any object mutation invalidates the
    // proxy scene (BVH rebuild), which ensurePrepared() picks up lazily
    ['OBJECT_LOADED', 'OBJECT_REMOVED', 'OBJECT_CHANGE'].forEach((name) => {
        K3D.on(K3D.events[name], () => {
            sceneDirty = true;
        });
    });

    // the rAF chain pauses in cinematic (a frame is a whole accumulation), so
    // camera interaction restarts the accumulation explicitly
    K3D.on(K3D.events.CAMERA_CHANGE, () => {
        if (K3D.parameters.renderer === 'cinematic') {
            generation++;
            K3D.render(true);
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

        env.mapping = THREE.EquirectangularReflectionMapping;
        target.environment = env;
        target.background = env;
        target.environmentRotation.copy(rotation);
        target.backgroundRotation.copy(rotation);
        target.environmentIntensity = K3D.parameters.lighting / 1.5;
        target.backgroundIntensity = K3D.parameters.lighting / 1.5;
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
        } else {
            node.textContent = text;
            node.style.display = 'block';
        }
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

    function renderUntil(target, gen, budget, present) {
        return new Promise((resolve, reject) => {
            const started = performance.now();

            function step() {
                if (gen !== generation) {
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
                    // GL queues work asynchronously - without a fence the
                    // wall time measures submission, not tracing
                    renderer.getContext().finish();
                    resolve({ samples, ms: performance.now() - started });
                    return;
                }

                setTimeout(step, 0);
            }

            step();
        });
    }

    // async by necessity: the library skips samples while its shaders compile
    // in the background (KHR_parallel_shader_compile), so the loop must yield
    // to the event loop until the accumulator reaches the budget
    function renderSamplesAsync(count, budget) {
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

            return renderUntil(count, gen, budget, budget > 0);
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

            ensurePrepared();
            backend.updateCamera();

            return renderSamplesAsync(budget, budget).then((result) => {
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

            ensurePrepared();
            backend.updateCamera();
            backend.setFixedSize(width, height);

            return renderSamplesAsync(budget, budget).then((result) => ({
                samples: result.samples,
                stale: result.stale,
                texture: backend.targetTexture(),
            }));
        },

        releaseFixedSize() {
            backend.releaseFixedSize();
            backend.reset();
        },

        // building blocks for headless probes and benchmarks
        renderSample() {
            return backend.renderSample();
        },

        renderSamplesAsync(count) {
            // probes hash the canvas, so the accumulation must reach it
            return renderSamplesAsync(count, count);
        },

        hideHud() {
            if (hud !== null) {
                hud.style.display = 'none';
            }
        },

        // strands any in-flight accumulation without touching prepared state
        abort() {
            generation++;
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
