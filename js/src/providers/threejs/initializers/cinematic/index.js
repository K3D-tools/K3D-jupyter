// Cinematic mode orchestration (stage 1 of renderer_cinematic.md): a filtered
// mirror of K3DObjects (only path-traceable materials), the environment as the
// sole light, an interruptible accumulation loop and a sample counter HUD.
// The full scene proxy (points -> merged icospheres, lines -> tubes) is stage 2,
// screenshots and tone mapping integration are stage 3.
const THREE = require('three');
const createWebGLBackend = require('./webglBackend');
const { getEnvironmentTexture } = require('../../helpers/environment');

module.exports = function cinematic(K3D, renderer) {
    const backend = createWebGLBackend(renderer);
    let ready = false;
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
            K3D.parameters.lighting,
        ].join('|');
    }

    function buildScene() {
        const world = K3D.getWorld();
        const scene = new THREE.Scene();

        world.K3DObjects.traverse((obj) => {
            // MaterialsTexture reads material.color.r unguarded, and only
            // Standard/Physical trace - filter by material, not object type
            if (obj.isMesh && obj.material
                && obj.material.color !== undefined
                && (obj.material.isMeshStandardMaterial || obj.material.isMeshPhysicalMaterial)
                && obj.visible) {
                const clone = obj.clone();

                clone.matrixAutoUpdate = false;
                clone.matrix.copy(obj.matrixWorld);
                clone.matrixWorld.copy(obj.matrixWorld);
                scene.add(clone);
            }
        });

        const env = getEnvironmentTexture(K3D.parameters.environment);

        env.mapping = THREE.EquirectangularReflectionMapping;
        scene.environment = env;
        scene.background = env;
        scene.environmentIntensity = K3D.parameters.lighting / 1.5;

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

        if (sceneDirty || key !== envKey) {
            setHud('cinematic: building BVH…');
            backend.setScene(buildScene(), K3D.getWorld().camera);
            sceneDirty = false;
            envKey = key;
            needsWarmup = true;
        }
    }

    function renderUntil(target, gen, budget) {
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
            ? renderUntil(1, gen, 0)
            : Promise.resolve({ samples: 0 });

        // rewind to the canonical RNG state before every accumulation - after
        // the warm-up when the sample texture just rebuilt, directly otherwise
        return warmup.then((first) => {
            if (first.stale) {
                return first;
            }

            needsWarmup = false;
            backend.reset();

            return renderUntil(count, gen, budget);
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

        // building blocks for headless probes and benchmarks
        renderSample() {
            return backend.renderSample();
        },

        renderSamplesAsync(count) {
            return renderSamplesAsync(count, 0);
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
            sceneDirty = true;
        },

        updateCamera: backend.updateCamera,
        reset: backend.reset,
        dispose: backend.dispose,
    };
};
