// Open Image Denoise over a converged accumulation, through oidn-web.
//
// The network runs on WebGPU; the tracer stays on WebGL. The two share nothing on the GPU, so
// the traced image crosses through the CPU: read back once, denoised in tiles, uploaded once.
// That is one round trip per converged frame, never per sample.
// the bundled build: lib/ is ESM with extensionless imports webpack refuses in a module package
const { initUNetFromBuffer } = require('oidn-web/dist/oidn.js');
const oidnWeightsSource = require('../../../../core/lib/oidnWeightsSource');

// Fixed, not adaptive: the padding below has to know the tile the library will cut.
const TILE = 512;
const ALIGN = 16;
// the larger receptive field of the two OIDN topologies, should a model not report its own
const FALLBACK_RECEPTIVE_FIELD = 202;

const MODELS = {
    aux: 'rt_hdr_alb_nrm.tza',
    plain: 'rt_hdr.tza',
};

function alignUp(value) {
    return Math.ceil(value / ALIGN) * ALIGN;
}

// oidn-web 0.4.0 reads a square source tile of tile + 2 * overlap without clipping it to the
// image, so any image narrower than that in either direction is read past its end and comes
// back NaN - every pixel, since the exposure is averaged over the tile. Padding up to the tile
// the library will cut keeps every read inside the image.
function paddedSize(width, height, receptiveField) {
    if (width <= TILE && height <= TILE) {
        const square = Math.max(alignUp(width), alignUp(height));

        return { width: square, height: square };
    }

    const overlap = alignUp(receptiveField / 2);
    const source = TILE + 2 * overlap;

    return { width: Math.max(width, source), height: Math.max(height, source) };
}

// edge-replicated copy of an RGBA image into a larger one
function pad(data, width, height, toWidth, toHeight, Type) {
    if (toWidth === width && toHeight === height) {
        return data;
    }

    const out = new Type(toWidth * toHeight * 4);

    for (let y = 0; y < toHeight; y++) {
        const sy = Math.min(y, height - 1);

        out.set(data.subarray(sy * width * 4, (sy + 1) * width * 4), y * toWidth * 4);

        const last = (sy * width + width - 1) * 4;

        for (let x = width; x < toWidth; x++) {
            out.set(data.subarray(last, last + 4), (y * toWidth + x) * 4);
        }
    }

    return out;
}

function crop(data, fromWidth, width, height) {
    if (fromWidth === width) {
        return data.subarray(0, width * height * 4);
    }

    const out = new Float32Array(width * height * 4);

    for (let y = 0; y < height; y++) {
        out.set(data.subarray(y * fromWidth * 4, (y * fromWidth + width) * 4), y * width * 4);
    }

    return out;
}

module.exports = function createOIDN() {
    const unets = {};
    let abort = null;
    let unavailable = null;
    // one network, one pass at a time: the preview and a screenshot can both ask
    let queue = Promise.resolve();

    function unavailableReason() {
        if (unavailable !== null) {
            return unavailable;
        }

        if (typeof navigator === 'undefined' || !navigator.gpu) {
            unavailable = 'this browser has no WebGPU';
        }

        return unavailable;
    }

    function loadUNet(kind) {
        if (!unets[kind]) {
            unets[kind] = oidnWeightsSource.read(MODELS[kind]).then((weights) => {
                if (weights === null) {
                    throw new Error(`the denoiser weights ${MODELS[kind]} could not be read`);
                }

                return initUNetFromBuffer(weights, undefined, {
                    hdr: true,
                    aux: kind === 'aux',
                    maxTileSize: TILE,
                    dynamicTile: false,
                });
            });
            // a failed load is not cached: the next converged frame may find the file
            unets[kind].catch(() => {
                delete unets[kind];
            });
        }

        return unets[kind];
    }

    // resolves with the denoised floats, or null when cancelled
    function execute(unet, color, albedo, normal, width, height) {
        return new Promise((resolve) => {
            const stop = unet.tileExecute({
                color: { data: color, width, height },
                ...(albedo ? {
                    albedo: { data: albedo, width, height },
                    normal: { data: normal, width, height },
                } : {}),
                done(output) {
                    abort = null;
                    resolve(output.data);
                },
            });

            abort = () => {
                stop();
                resolve(null);
            };
        });
    }

    return {
        unavailableReason,

        // color: premultiplied linear RGBA floats; albedo and normal: RGBA bytes, or both null.
        // Resolves with denoised premultiplied RGBA floats of the same size, or null if cancelled.
        denoise(color, albedo, normal, width, height) {
            const kind = albedo ? 'aux' : 'plain';
            const run = () => loadUNet(kind).then((unet) => {
                const field = (unet._modelSpec && unet._modelSpec.receptiveField)
                    || FALLBACK_RECEPTIVE_FIELD;
                const size = paddedSize(width, height, field);
                const rgb = new Float32Array(width * height * 4);
                let fractionalAlpha = 0;

                for (let i = 0; i < width * height * 4; i += 4) {
                    // a single non-finite input poisons the exposure of its whole tile
                    rgb[i] = Number.isFinite(color[i]) ? Math.max(color[i], 0) : 0;
                    rgb[i + 1] = Number.isFinite(color[i + 1]) ? Math.max(color[i + 1], 0) : 0;
                    rgb[i + 2] = Number.isFinite(color[i + 2]) ? Math.max(color[i + 2], 0) : 0;
                    rgb[i + 3] = 1;

                    if (color[i + 3] > 0.02 && color[i + 3] < 0.98) {
                        fractionalAlpha++;
                    }
                }

                const pw = size.width;
                const ph = size.height;
                const paddedAlbedo = albedo ? pad(albedo, width, height, pw, ph, Uint8ClampedArray) : null;
                const paddedNormal = normal ? pad(normal, width, height, pw, ph, Uint8ClampedArray) : null;

                return execute(
                    unet,
                    pad(rgb, width, height, pw, ph, Float32Array),
                    paddedAlbedo,
                    paddedNormal,
                    pw,
                    ph,
                ).then((denoised) => {
                    if (denoised === null) {
                        return null;
                    }

                    const out = crop(denoised, pw, width, height);

                    // Coverage is traced too - a medium thins out stochastically - so where it
                    // is fractional its grain is denoised the same way as the colour's.
                    if (fractionalAlpha < width * height * 0.001) {
                        for (let i = 3; i < width * height * 4; i += 4) {
                            out[i] = Math.min(Math.max(color[i], 0), 1);
                        }

                        return out;
                    }

                    const alpha = new Float32Array(width * height * 4);

                    for (let i = 0; i < width * height * 4; i += 4) {
                        const a = Number.isFinite(color[i + 3]) ? Math.min(Math.max(color[i + 3], 0), 1) : 0;

                        alpha[i] = a;
                        alpha[i + 1] = a;
                        alpha[i + 2] = a;
                        alpha[i + 3] = 1;
                    }

                    return execute(
                        unet,
                        pad(alpha, width, height, pw, ph, Float32Array),
                        paddedAlbedo,
                        paddedNormal,
                        pw,
                        ph,
                    ).then((denoisedAlpha) => {
                        if (denoisedAlpha === null) {
                            return null;
                        }

                        const a = crop(denoisedAlpha, pw, width, height);

                        for (let i = 0; i < width * height * 4; i += 4) {
                            out[i + 3] = Math.min(Math.max(a[i], 0), 1);
                        }

                        return out;
                    });
                });
            });
            const result = queue.then(run);

            // a failed pass must not stall the next one
            queue = result.catch(() => null);

            return result;
        },

        cancel() {
            if (abort !== null) {
                abort();
                abort = null;
            }
        },
    };
};

module.exports.paddedSize = paddedSize;
