const THREE = require('three');
const fflate = require('fflate');
const Float16Array = require('../../../core/lib/helpers/float16Array');
const buffer = require('../../../core/lib/helpers/buffer');

const WIRE_TYPES = {
    int8: Int8Array,
    int16: Int16Array,
    int32: Int32Array,
    uint16: Uint16Array,
    uint32: Uint32Array,
    float32: Float32Array,
    float64: Float64Array,
};

function halfToFloat(h) {
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;

    if (e === 0) {
        return s * m * 2 ** -24;
    }
    if (e === 31) {
        return m ? NaN : s * Infinity;
    }

    return s * (1024 + m) * 2 ** (e - 25);
}

// The widget path deserializes typed arrays, but the headless/snapshot plot-parameter
// path hands them over as raw msgpack bytes - reinterpret by the declared dtype.
// slice() first: the bytes can sit unaligned inside the message buffer. float16 is
// decoded numerically, because the Float16Array shim only relabels raw bits.
function ensureTyped(env) {
    const isBytes = env.data instanceof Uint8Array && env.dtype !== 'uint8';

    if (env.dtype === 'float16' && (isBytes || env.data.constructor === Float16Array)) {
        const bits = isBytes ? new Uint16Array(env.data.slice().buffer) : env.data;
        const data = new Float32Array(bits.length);

        for (let i = 0; i < bits.length; i++) {
            data[i] = halfToFloat(bits[i]);
        }

        return { shape: env.shape, data };
    }

    const Type = WIRE_TYPES[env.dtype];

    if (Type && isBytes) {
        return { shape: env.shape, data: new Type(env.data.slice().buffer) };
    }

    return env;
}

// Generated on the CPU so the result is bit-identical on every GPU - the test suite
// depends on that determinism. 256x128: the outdoor sun disc must span a few texels,
// at 128x64 it aliased away.
const WIDTH = 256;
const HEIGHT = 128;

// Mean irradiance every map is normalised to. The sphere-average equivalent of the replaced
// rig is 0.4 PI (ambient 0.2 PI in full + key/head/fill/back 0.8 PI at the 1/4 directional
// factor), but the old rig chased the camera, favouring the surfaces one actually sees -
// calibrated against the full reference base instead: median advanced/simple scene
// brightness at 0.4 PI was 0.894, hence the 1/0.894 correction. Measured back at 0.925, not
// 1.0 - bright scenes saturate, so the response is sublinear; pushing the median further
// overexposes volumetrics (already at 1.0-1.1). In advanced the environment is the only
// light.
const TARGET_IRRADIANCE = 0.45 * Math.PI;

function gauss(x, sigma) {
    return Math.exp(-(x * x) / (2.0 * sigma * sigma));
}

// three's equirectUv maps v = 0 to the BOTTOM pole (dir.y = -1), so v grows upwards here.
const PRESETS = {
    neutral(u, v) {
        // The gradient alone leaves the specular lobe with nothing to reflect - dark
        // glossy surfaces render as silhouettes. An achromatic key softbox (azimuth
        // aimed at the default camera) plus a weaker counter-fill; the normalisation
        // pays for them out of the gradient, so diffuse exposure stays put.
        let l = 0.5 + 0.9 * v;

        l += 4.0 * gauss(u - 0.33, 0.1) * gauss(v - 0.72, 0.08);
        l += 1.2 * gauss(u - 0.78, 0.18) * gauss(v - 0.62, 0.12);

        return [l, l, l];
    },

    // a dark shell with hard, cool softboxes off to the side: the contrast and the
    // shading direction are the point - the energy normalisation keeps the exposure,
    // so what changes is the drama of the shadows, not the brightness
    studio(u, v) {
        let l = 0.1 + 0.2 * v;

        l += 6.0 * gauss(u - 0.55, 0.05) * gauss(v - 0.72, 0.08);
        l += 2.0 * gauss(u - 0.05, 0.08) * gauss(v - 0.6, 0.12);
        l += 1.2 * gauss(v - 0.98, 0.03);

        return [l * 0.9, l, l * 1.18];
    },

    // a clear sunny day: deep blue sky, a hard warm sun disc with a halo, earthy
    // bounce - blue ambient in the shadows, warm key on lit faces
    outdoor(u, v) {
        if (v > 0.5) {
            // the sun carries most of a clear day's energy and sits low enough
            // (afternoon) to light vertical faces; the sky is the blue of the
            // shadows, not of the whole image
            const t = (v - 0.5) * 2.0;
            const sun = 120.0 * gauss(u - 0.33, 0.02) * gauss(v - 0.72, 0.03);
            const halo = 5.0 * gauss(u - 0.33, 0.07) * gauss(v - 0.72, 0.09);
            const warm = sun + halo;

            return [
                0.45 + 0.1 * (1.0 - t) + warm,
                0.58 + 0.1 * (1.0 - t) + warm * 0.85,
                1.0 + 0.3 * t + warm * 0.55,
            ];
        }
        const ground = 0.45 - 0.5 * (0.5 - v);

        return [ground * 1.15, ground * 0.9, ground * 0.6];
    },
};

// Solid-angle weighted mean radiance of an equirect, times PI = irradiance of a white
// lambertian under it. Applied to user maps as well: environment carries the shape of the
// light, the lighting knob carries the exposure.
function normalise(data, width, height) {
    let sum = 0.0;
    let weightSum = 0.0;

    for (let y = 0; y < height; y++) {
        const w = Math.sin((Math.PI * (y + 0.5)) / height);

        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;

            sum += (w * (data[i] + data[i + 1] + data[i + 2])) / 3.0;
            weightSum += w;
        }
    }

    const scale = TARGET_IRRADIANCE / (Math.PI * (sum / weightSum));

    // Radiance projected onto the 9 spherical harmonics, in three's basis order - the
    // bespoke-light shaders (volume, mip, points 3d) evaluate it with shGetIrradianceAt,
    // which applies the irradiance band constants itself.
    const sh = new Float32Array(27);
    const basis = new Float32Array(9);

    for (let y = 0; y < height; y++) {
        const lat = Math.PI * ((y + 0.5) / height - 0.5);
        const cosLat = Math.cos(lat);
        const dy = Math.sin(lat);
        const dOmega = ((2.0 * Math.PI) / width) * (Math.PI / height) * cosLat;

        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;

            data[i] *= scale;
            data[i + 1] *= scale;
            data[i + 2] *= scale;

            const phi = 2.0 * Math.PI * ((x + 0.5) / width - 0.5);
            const dx = cosLat * Math.cos(phi);
            const dz = cosLat * Math.sin(phi);

            basis[0] = 0.282095;
            basis[1] = 0.488603 * dy;
            basis[2] = 0.488603 * dz;
            basis[3] = 0.488603 * dx;
            basis[4] = 1.092548 * dx * dy;
            basis[5] = 1.092548 * dy * dz;
            basis[6] = 0.315392 * (3.0 * dz * dz - 1.0);
            basis[7] = 1.092548 * dx * dz;
            basis[8] = 0.546274 * (dx * dx - dy * dy);

            for (let b = 0; b < 9; b++) {
                const w = basis[b] * dOmega;

                sh[b * 3] += data[i] * w;
                sh[b * 3 + 1] += data[i + 1] * w;
                sh[b * 3 + 2] += data[i + 2] * w;
            }
        }
    }

    return { data, sh };
}

function equirectTexture(data, width, height) {
    const texture = new THREE.DataTexture(
        data,
        width,
        height,
        THREE.RGBAFormat,
        THREE.FloatType,
    );

    texture.mapping = THREE.EquirectangularReflectionMapping;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;

    return texture;
}

function fromPreset(name) {
    const preset = PRESETS[name] || PRESETS.neutral;

    if (!PRESETS[name]) {
        console.warn(`K3D: unknown environment "${name}", falling back to "neutral"`);
    }

    const data = new Float32Array(WIDTH * HEIGHT * 4);

    for (let y = 0; y < HEIGHT; y++) {
        const v = (y + 0.5) / HEIGHT;

        for (let x = 0; x < WIDTH; x++) {
            const u = (x + 0.5) / WIDTH;
            const c = preset(u, v);
            const i = (y * WIDTH + x) * 4;

            data[i] = c[0];
            data[i + 1] = c[1];
            data[i + 2] = c[2];
            data[i + 3] = 1.0;
        }
    }

    const normalised = normalise(data, WIDTH, HEIGHT);
    const texture = equirectTexture(normalised.data, WIDTH, HEIGHT);

    texture.userData.k3dSH = normalised.sh;

    return texture;
}

// A user map arrives as an (H, W, 3) float32 array, already decoded by the Python side.
// Image row 0 is the sky; the texture's row 0 samples at v = 0, the bottom pole - flip.
function fromArray(env) {
    const height = env.shape[0];
    const width = env.shape[1];
    const data = new Float32Array(width * height * 4);

    for (let y = 0; y < height; y++) {
        const src = (height - 1 - y) * width * 3;
        const dst = y * width * 4;

        for (let x = 0; x < width; x++) {
            data[dst + x * 4] = env.data[src + x * 3];
            data[dst + x * 4 + 1] = env.data[src + x * 3 + 1];
            data[dst + x * 4 + 2] = env.data[src + x * 3 + 2];
            data[dst + x * 4 + 3] = 1.0;
        }
    }

    const normalised = normalise(data, width, height);
    const texture = equirectTexture(normalised.data, width, height);

    texture.userData.k3dSH = normalised.sh;

    return texture;
}

// The photographic catalog lives in the python package, so a kernel-less page
// degrades its names to neutral - unless a sideload provides the maps: a shared
// k3dEnvironments.js (k3d.environments.save_js) defines window.k3dEnvironments
// as {name: {b64: base64(zlib(float16 bytes)), shape: [h, w, 3]}}. Decoded once,
// cached on the entry.
function fromSideload(name) {
    const catalog = (typeof (window) !== 'undefined') && window.k3dEnvironments;

    if (!catalog || !catalog[name]) {
        return null;
    }

    const entry = catalog[name];

    if (!entry.decoded) {
        entry.decoded = ensureTyped({
            data: fflate.unzlibSync(new Uint8Array(buffer.base64ToArrayBuffer(entry.b64))),
            dtype: 'float16',
            shape: entry.shape,
        });
    }

    return fromArray(entry.decoded);
}

module.exports = {
    getEnvironmentTexture(environment) {
        if (environment && environment.data && environment.shape) {
            return fromArray(ensureTyped(environment));
        }

        const name = typeof (environment) === 'string' ? environment : 'neutral';

        return fromSideload(name) || fromPreset(name);
    },
};
