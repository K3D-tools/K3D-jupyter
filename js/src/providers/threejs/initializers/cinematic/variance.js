// How far from converged the traced image is, per pixel, for a filter to be guided by.
//
// Split the accumulation in two by sample parity and the spread between the halves IS the noise:
// Var(A - B) = sigma^2 (1/nA + 1/nB) and Var(M) = sigma^2 / n, so the variance of the mean the
// viewer is looking at is (A - B)^2 * nA * nB / n^2, with no second estimate of sigma needed.
//
// It costs nothing in the tracing shader, which is the whole reason for this shape. K3D always
// runs the tracer's alpha path (index.js sets scene.background to null, so backgroundAlpha is 0
// and PathTracingRenderer takes its manual-blend branch), and there the tracer writes each sample
// RAW into _primaryTarget and averages it in a separate pass. So the per-sample image is already
// a texture, and the halves are two more blends of a texture that exists - no second program
// variant, and none of the ~16 s of shader compilation one would cost.
//
// The blend is upstream's own BlendMaterial rather than a copy of it, so the halves weight alpha
// exactly the way the image they measure does.
//
// Nothing is read back to the CPU. An earlier version reduced the halves to a single noise figure
// for the HUD, which meant an asynchronous readback holding a PIXEL_PACK buffer - and while that
// buffer is bound, every synchronous readRenderTargetPixels in the process fails, including the
// one the screenshot path depends on, so a screenshot taken while it was alive came back empty.
// That figure was also normalised against the whole frame, which overstates the noise on any
// subject that does not fill it. The filter reads these textures in its own shader and never
// needed either.
const THREE = require('three');
const { FullScreenQuad } = require('three/examples/jsm/postprocessing/Pass.js');
const { BlendMaterial } = require('three-gpu-pathtracer/src/materials/fullscreen/BlendMaterial.js');

function makeTarget(width, height) {
    return new THREE.WebGLRenderTarget(width, height, {
        format: THREE.RGBAFormat,
        // not half float: this is a difference of nearly equal numbers, and fp16 would put a floor
        // under it exactly where the image is close to converged and the filter should back off
        type: THREE.FloatType,
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        depthBuffer: false,
        stencilBuffer: false,
        generateMipmaps: false,
    });
}

module.exports = function createVarianceBuffer(renderer) {
    let enabled = false;
    let unusable = null;
    let width = 0;
    let height = 0;
    // halves[0] takes the odd samples, halves[1] the even ones; scratch rotates in because
    // BlendMaterial cannot read and write one texture, so three targets carry two halves
    let halves = null;
    let scratch = null;
    let counts = [0, 0];
    let blendQuad = null;
    let lastWhole = 0;
    let lastEpoch = -1;

    function freeTargets() {
        if (halves === null) {
            return;
        }

        halves[0].dispose();
        halves[1].dispose();
        scratch.dispose();
        halves = null;
        scratch = null;
    }

    function clearHalves() {
        const previous = renderer.getRenderTarget();

        halves.forEach((target) => {
            renderer.setRenderTarget(target);
            renderer.clear(true, false, false);
        });
        renderer.setRenderTarget(previous);

        counts = [0, 0];
        lastWhole = 0;
    }

    function allocate(w, h) {
        freeTargets();

        width = w;
        height = h;
        halves = [makeTarget(w, h), makeTarget(w, h)];
        scratch = makeTarget(w, h);

        clearHalves();
    }

    // The tracer saves and restores its own render state around each tile, but it is mid-flight
    // between our calls, so everything we touch goes back exactly as it was.
    function withRenderState(fn) {
        const previousTarget = renderer.getRenderTarget();
        const previousAutoClear = renderer.autoClear;
        const previousScissorTest = renderer.getScissorTest();
        const scissor = new THREE.Vector4();
        const viewport = new THREE.Vector4();

        renderer.getScissor(scissor);
        renderer.getViewport(viewport);
        renderer.autoClear = false;
        renderer.setScissorTest(false);

        try {
            fn();
        } finally {
            renderer.setRenderTarget(previousTarget);
            renderer.autoClear = previousAutoClear;
            renderer.setScissorTest(previousScissorTest);
            renderer.setScissor(scissor);
            renderer.setViewport(viewport);
        }
    }

    return {
        setEnabled(value) {
            const next = Boolean(value) && unusable === null;

            if (next === enabled) {
                return;
            }

            enabled = next;

            if (!enabled) {
                freeTargets();
                width = 0;
                height = 0;
            }
        },

        isEnabled() {
            return enabled;
        },

        // Called after every renderSample, which is once per TILE. Everything below is gated on a
        // whole sample having just landed, so the per-tile cost is one comparison.
        afterRenderSample(pathTracer, samples, epoch) {
            if (!enabled || unusable !== null) {
                return;
            }

            const primary = pathTracer && pathTracer._primaryTarget;

            if (!primary || pathTracer._alpha !== true) {
                // Without the alpha path _primaryTarget is blended in place and holds the mean
                // rather than the sample, and the estimator is meaningless. Refuse rather than
                // hand a filter a guide computed from a mean.
                unusable = 'the tracer is not running its alpha path';
                enabled = false;
                freeTargets();

                return;
            }

            if (primary.width !== width || primary.height !== height) {
                allocate(primary.width, primary.height);
            }

            // The counter alone cannot see every reset: at one tile a reset from sample 1 back
            // to 1 leaves it unchanged. The epoch is bumped by everything that resets.
            if (epoch !== lastEpoch || samples < lastWhole) {
                lastEpoch = epoch;
                clearHalves();
            }

            const whole = Math.round(samples);

            if (Math.abs(samples - whole) > 1e-6 || whole <= lastWhole) {
                return;
            }

            if (blendQuad === null) {
                blendQuad = new FullScreenQuad(new BlendMaterial());
            }

            // odd samples to half 0, even to half 1
            const p = (whole - 1) % 2;

            counts[p] += 1;
            lastWhole = whole;

            withRenderState(() => {
                const { material } = blendQuad;

                material.target1 = halves[p].texture;
                material.target2 = primary.texture;
                material.opacity = 1.0 / counts[p];

                renderer.setRenderTarget(scratch);
                blendQuad.render(renderer);

                material.target1 = null;
                material.target2 = null;

                const written = scratch;

                scratch = halves[p];
                halves[p] = written;
            });
        },

        reset() {
            if (enabled && halves !== null) {
                clearHalves();
            }
        },

        // The two halves and the factor that turns their difference into the variance of the mean.
        // Null until both have received a sample: one alone says nothing about spread.
        halves() {
            if (!enabled || halves === null || counts[0] === 0 || counts[1] === 0) {
                return null;
            }

            const n = counts[0] + counts[1];

            return {
                a: halves[0].texture,
                b: halves[1].texture,
                weight: (counts[0] * counts[1]) / (n * n),
                samples: n,
            };
        },

        unusableReason() {
            return unusable;
        },

        dispose() {
            freeTargets();

            if (blendQuad !== null) {
                blendQuad.material.dispose();
                blendQuad.dispose();
                blendQuad = null;
            }
        },
    };
};
