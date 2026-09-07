// The path tracing material with one sampled volume: sceneProxy puts K3D's Volume into the BVH
// as a closed box with FogVolumeMaterial, so upstream handles entering and leaving the medium,
// and this material replaces the homogeneous fog inside with delta tracking over the Data3DTexture
// that the raster Volume object already owns.
const THREE = require('three');
const {
    PhysicalPathTracingMaterial,
} = require('three-gpu-pathtracer/src/materials/pathtracing/PhysicalPathTracingMaterial.js');
const RenderGLSL = require('three-gpu-pathtracer/src/materials/pathtracing/glsl/index.js');
const BSDFGLSL = require('three-gpu-pathtracer/src/shader/bsdf/index.js');
const glsl = require('./glsl');
const { MajorantGrid } = require('./majorant');

// upstream's fragment shader holds 15 samplers with K3D's background off, less the two the
// dead light path used; the volume adds its data and its majorant grid
const REQUIRED_TEXTURE_UNITS = 15;
// the packed transfer function on top of what upstream already declares
const REQUIRED_UNIFORM_VECTORS = glsl.TF_SIZE / 4 + 128;

function internalsChanged(what) {
    return new Error(`cinematic: three-gpu-pathtracer internals changed - ${what}`);
}

// the chunk has to appear exactly once, verbatim, or the replacement would silently misfire
function replaceOnce(source, needle, replacement, what) {
    const first = source.indexOf(needle);

    if (first === -1 || source.indexOf(needle, first + 1) !== -1) {
        throw internalsChanged(what);
    }

    // the function form keeps "$" sequences in GLSL out of the replacement pattern syntax
    return source.replace(needle, () => replacement);
}

// No K3D object builds a MeshPhysicalMaterial, so sheen and iridescence are zero for every
// material - but neither is guarded, so both are evaluated on every bounce and then multiplied by
// that zero. The call sites come out before the chunks: their callers sit in bsdf_functions,
// which this file re-inserts verbatim, so dropping the chunks alone would not compile.
const IRIDESCENCE_CALL = '\n\t\tvec3 iridescenceF = evalIridescence( 1.0, surf.iridescenceIor,'
    + ' dot( wi, wh ), surf.iridescenceThickness, f0Color );\n'
    + '\t\tF = mix( F, iridescenceF,  surf.iridescence );\n';

const SHEEN_CALLS = '\n\t\t// sheen\n'
    + '\t\tcolor *= mix( 1.0, sheenAlbedoScaling( wo, wi, surf ), surf.sheen );\n'
    + '\t\tcolor += sheenColor( wo, wi, halfVector, surf ) * surf.sheen;\n';

const SHEEN_COLOR_FN = '\n\t// sheen\n'
    + '\tvec3 sheenColor( vec3 wo, vec3 wi, vec3 wh, SurfaceRecord surf ) {\n\n'
    + '\t\tfloat cosThetaO = saturateCos( wo.z );\n'
    + '\t\tfloat cosThetaI = saturateCos( wi.z );\n'
    + '\t\tfloat cosThetaH = wh.z;\n\n'
    + '\t\tfloat D = velvetD( cosThetaH, surf.sheenRoughness );\n'
    + '\t\tfloat G = velvetG( cosThetaO, cosThetaI, surf.sheenRoughness );\n\n'
    + '\t\t// See equation (1) in http://www.aconty.com/pdf/s2017_pbs_imageworks_sheen.pdf\n'
    + '\t\tvec3 color = surf.sheenColor;\n'
    + '\t\tcolor *= D * G / ( 4.0 * abs( cosThetaO * cosThetaI ) );\n'
    + '\t\tcolor *= wi.z;\n\n'
    + '\t\treturn color;\n\n'
    + '\t}\n';

function stripDeadLobes(source) {
    let shader = source;

    shader = replaceOnce(
        shader,
        IRIDESCENCE_CALL,
        '',
        'the iridescence call in specularEval is not in the fragment shader exactly once',
    );

    shader = replaceOnce(
        shader,
        SHEEN_CALLS,
        '',
        'the sheen calls in bsdfEval is not in the fragment shader exactly once',
    );

    shader = replaceOnce(
        shader,
        SHEEN_COLOR_FN,
        '',
        'the sheenColor definition is not in the fragment shader exactly once',
    );

    shader = replaceOnce(
        shader,
        BSDFGLSL.iridescence_functions,
        '',
        'the iridescence chunk is not in the fragment shader exactly once',
    );

    shader = replaceOnce(
        shader,
        BSDFGLSL.sheen_functions,
        '',
        'the sheen chunk is not in the fragment shader exactly once',
    );

    return shader;
}

function patchFragmentShader(source) {
    let shader = source;

    // declarations go right before the BSDF chunk, the earliest consumer (the phase function)
    shader = replaceOnce(
        shader,
        BSDFGLSL.bsdf_functions,
        `${glsl.volumeDeclarations}\n#define bsdfSample bsdfSampleUpstream\n#define bsdfResult bsdfResultUpstream\n`
        + `${BSDFGLSL.bsdf_functions}\n#undef bsdfSample\n#undef bsdfResult\n${glsl.volumeBsdf}`,
        'bsdf chunk is not in the fragment shader exactly once',
    );

    shader = replaceOnce(
        shader,
        RenderGLSL.trace_scene_function,
        '#define traceScene traceSceneUpstream\n'
        + `${RenderGLSL.trace_scene_function}\n#undef traceScene\n${glsl.volumeTraceScene}`,
        'traceScene chunk is not in the fragment shader exactly once',
    );

    shader = replaceOnce(
        shader,
        '\t\t\t\tuniform sampler2DArray iesProfiles;\n'
        + '\t\t\t\tuniform LightsInfo lights;',
        glsl.volumeNoLights,
        'the light uniforms are not in the fragment shader exactly once',
    );

    shader = replaceOnce(
        shader,
        RenderGLSL.direct_light_contribution_function,
        `${glsl.volumeNoLightMacros}#define directLightContribution directLightContributionUpstream\n`
        + `${RenderGLSL.direct_light_contribution_function}\n#undef directLightContribution\n`
        + `${glsl.volumeDirectLight}`,
        'directLightContribution chunk is not in the fragment shader exactly once',
    );

    // the branch that samples a light object: dead with no lights in the scene, and the
    // compiler folds its weight by a zero pdf into a division by zero it then reports
    shader = replaceOnce(
        shader,
        '\t\tif( lightsDenom != 0.0 && rand( 5 ) < float( lights.count ) / lightsDenom ) {',
        '#if 0\n\t\tif ( false ) {',
        'the light branch of directLightContribution is not in the fragment shader exactly once',
    );

    shader = replaceOnce(
        shader,
        '\t\t} else if ( envMapInfo.totalSum != 0.0 && environmentIntensity != 0.0 ) {',
        '\t\t}\n#endif\n\t\tif ( envMapInfo.totalSum != 0.0 && environmentIntensity != 0.0 ) {',
        'the environment branch of directLightContribution is not in the fragment shader exactly once',
    );

    // the other half of the exposure: what a path reaches after a medium event without
    // another event is the same estimator next event estimation samples, so it carries the
    // same factor or the MIS pair disagrees
    shader = replaceOnce(
        shader,
        'gl_FragColor.rgb += environmentIntensity * envColor * state.throughputColor * misWeight;',
        'gl_FragColor.rgb += environmentIntensity * envColor * state.throughputColor * misWeight'
        + ' * k3dVolumeLightScale();',
        'environment escape in main() is not in the fragment shader exactly once',
    );

    shader = replaceOnce(
        shader,
        'gl_FragColor.rgb += lightRec.emission * state.throughputColor * misWeight;',
        'gl_FragColor.rgb += lightRec.emission * state.throughputColor * misWeight'
        + ' * k3dVolumeLightScale();',
        'forward light hit in main() is not in the fragment shader exactly once',
    );

    shader = replaceOnce(
        shader,
        RenderGLSL.get_surface_record_function,
        `#define getSurfaceRecord getSurfaceRecordUpstream\n${RenderGLSL.get_surface_record_function}\n`
        + `#undef getSurfaceRecord\n${glsl.volumeSurfaceRecord}`,
        'getSurfaceRecord chunk is not in the fragment shader exactly once',
    );

    // after the upstream RNG setup in main(), so the stratified salt is ready
    shader = replaceOnce(
        shader,
        'sobolPathIndex = uint( seed );',
        `sobolPathIndex = uint( seed );\n${glsl.volumeRngInit}`,
        'RNG setup in main() is not in the fragment shader exactly once',
    );

    // a scattering event is a real interaction: the escaping ray takes the MIS branch and the
    // pixel keeps its alpha, instead of being treated as a straight-through transmissive ray.
    // MaterialsTexture writes the fog material with castShadow off, and after a diffuse bounce
    // upstream steps through every non-shadowing hit, so without the flag the path would ignore
    // the medium from its second collision on and only the first event would ever be lit
    shader = replaceOnce(
        shader,
        'state.accumulatedRoughness += 0.2;',
        'k3dVolumeClassify( ray.direction );\n'
        + 'state.accumulatedRoughness += k3dVolumeHitSurface ? 0.0 : 0.2;\n'
        + 'state.transmissiveRay = false;\n'
        + 'if ( k3dVolumeHit ) { material.castShadow = true; }',
        'fog hit handling in main() is not in the fragment shader exactly once',
    );

    return stripDeadLobes(shader);
}

// RGBA8 pixels of the row a texture( lut, vec2( x, 0.5 ) ) lookup reads; the K3D LUT is a
// 1024x1 CanvasTexture, a DataTexture with Uint8 or Float32 RGBA data is accepted as well
function readLutRow(texture) {
    const image = texture && texture.image;

    if (!image || !image.width || !image.height) {
        throw new Error('cinematic: volume transfer function texture has no image');
    }

    const { width, height } = image;
    let data = null;

    if (image.data) {
        data = image.data;
    } else if (typeof image.getContext === 'function') {
        data = image.getContext('2d').getImageData(0, 0, width, height).data;
    }

    if (!data || data.length < width * height * 4) {
        throw new Error('cinematic: volume transfer function texture has no readable RGBA pixels');
    }

    const row = Math.floor(height / 2) * width * 4;
    const pixels = new Uint8Array(width * 4);
    const isFloat = data instanceof Float32Array || data instanceof Float64Array;

    for (let i = 0; i < width * 4; i++) {
        const v = data[row + i];

        pixels[i] = isFloat ? Math.round(Math.min(Math.max(v, 0.0), 1.0) * 255.0) : v;
    }

    return { pixels, width };
}

// packs the LUT row into the uvec4 uniform array and finds the largest alpha for the majorant
function packTransferFunction(texture, packed) {
    const { pixels, width } = readLutRow(texture);
    const size = Math.min(width, glsl.TF_SIZE);
    let maxAlpha = 0;

    if (width > glsl.TF_SIZE) {
        console.warn(`K3D.cinematic: volume transfer function wider than ${glsl.TF_SIZE} texels is `
            + 'resampled for the path tracer');
    }

    packed.fill(0);

    for (let i = 0; i < size; i++) {
        const src = (width === size ? i : Math.floor((i * width) / size)) * 4;
        const alpha = pixels[src + 3];

        packed[i] = (pixels[src] | (pixels[src + 1] << 8) | (pixels[src + 2] << 16) | (alpha << 24)) >>> 0;
        maxAlpha = Math.max(maxAlpha, alpha);
    }

    return { size, maxAlpha: maxAlpha / 255.0 };
}

class VolumePathTracingMaterial extends PhysicalPathTracingMaterial {
    constructor(parameters) {
        super(parameters);

        Object.assign(this.uniforms, {
            volumeEnabled: { value: 0 },
            volumeTexture: { value: null },
            volumeTF: { value: new Uint32Array(glsl.TF_SIZE) },
            volumeTFSize: { value: glsl.TF_SIZE },
            volumeLow: { value: 0.0 },
            volumeHigh: { value: 1.0 },
            volumeAlphaCoef: { value: 0.0 },
            volumeSigmaMax: { value: 0.0 },
            volumeMajorant: { value: null },
            volumeMajorantCells: { value: new THREE.Vector3(1, 1, 1) },
            volumeMajorantScale: { value: new THREE.Vector3(1, 1, 1) },
            volumeInvMatrix: { value: new THREE.Matrix4() },
            volumeLightScale: { value: 1.0 },
            volumeRoughness: { value: 0.25 },
            volumeMetalness: { value: 0.0 },
            volumeGradientStep: { value: new THREE.Vector3(0.005, 0.005, 0.005) },
            volumeGradientScale: { value: 1.0 },
            volumeSurfaceK: { value: glsl.SURFACE_K },
            volumePhaseG: { value: glsl.PHASE_G },
        });

        // the transfer function is read back from a canvas, which the browser warns about when it
        // happens repeatedly, and setVolume runs after every setScene and every material edit
        this.majorant = new MajorantGrid();
        this.warnedDense = false;
        this.lut = {
            texture: null,
            version: -1,
            size: glsl.TF_SIZE,
            maxAlpha: 0,
        };

        // the medium is compiled in only while a volume is set, so a scene without one gets the
        // upstream program: no extra sampler, no tracking code
        this.defines.K3D_VOLUME = 0;
        this.fragmentShader = patchFragmentShader(this.fragmentShader);
    }

    // null when the volume path can compile on this context, else the reason
    static unsupportedReason(renderer) {
        const gl = renderer.getContext();
        const units = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
        const vectors = gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS);

        if (units < REQUIRED_TEXTURE_UNITS) {
            return `the path tracer with a volume needs ${REQUIRED_TEXTURE_UNITS} fragment texture units, `
                + `this context has ${units}`;
        }

        if (vectors < REQUIRED_UNIFORM_VECTORS) {
            return `the path tracer with a volume needs ${REQUIRED_UNIFORM_VECTORS} fragment uniform `
                + `vectors, this context has ${vectors}`;
        }

        return null;
    }

    // texture and transferFunction are the Volume object's own instances; matrixWorld is the
    // world matrix of its unit box; roughness, metalness and gradientStep are the Volume's own
    // traits, the rest are the hybrid's constants unless a caller overrides them
    setVolume({
        texture, transferFunction, low, high, alphaCoef, matrixWorld,
        roughness = 0.25, metalness = 0.0, gradientStep = 0.005, lightScale = 1.0,
        surfaceK = glsl.SURFACE_K, phaseG = glsl.PHASE_G,
    }, renderer) {
        const u = this.uniforms;

        if (this.lut.texture !== transferFunction || this.lut.version !== transferFunction.version) {
            const packed = packTransferFunction(transferFunction, u.volumeTF.value);

            this.lut = {
                texture: transferFunction,
                version: transferFunction.version,
                size: packed.size,
                maxAlpha: packed.maxAlpha,
            };
        }

        const lut = this.lut;
        // the headroom covers the float32 log in the shader against this float64 one
        const sigmaMax = glsl.alphaToSigma(alphaCoef, lut.maxAlpha) * (1.0 + 1e-5);

        // a NaN anywhere here would turn every ray into a full-length tracking loop
        if (![low, high, alphaCoef, sigmaMax, roughness, metalness, gradientStep, lightScale,
            surfaceK, phaseG]
            .every(Number.isFinite) || high === low) {
            console.warn('K3D.cinematic: volume has a non-finite range, alpha_coef or surface '
                + 'parameter, left to the raster layer');
            this.clearVolume();

            return;
        }

        u.volumeTexture.value = texture;
        u.volumeTFSize.value = lut.size;
        u.volumeLow.value = low;
        u.volumeHigh.value = high;
        u.volumeAlphaCoef.value = alphaCoef;
        u.volumeSigmaMax.value = sigmaMax;
        u.volumeInvMatrix.value.copy(matrixWorld).invert();
        u.volumeLightScale.value = Math.max(lightScale, 0.0);
        u.volumeRoughness.value = roughness;
        u.volumeMetalness.value = metalness;
        u.volumeSurfaceK.value = surfaceK;
        u.volumePhaseG.value = Math.min(Math.max(phaseG, -0.99), 0.99);

        // one gradient step is gradient_step of the mean box edge, in world units, so the finite
        // differences span the same distance on every axis whatever the voxel shape
        const size = new THREE.Vector3().setFromMatrixScale(matrixWorld);
        const stepWorld = (gradientStep * (size.x + size.y + size.z)) / 3.0;
        const image = texture.image || {};
        const voxel = new THREE.Vector3(
            size.x / Math.max(image.width || 1, 1),
            size.y / Math.max(image.height || 1, 1),
            size.z / Math.max(image.depth || 1, 1),
        );

        u.volumeGradientStep.value.set(stepWorld / size.x, stepWorld / size.y, stepWorld / size.z);
        // the differences span 2 * stepWorld; scaled to one mean voxel they become a rate per
        // voxel, which is the sharpness the surface probability is calibrated against
        u.volumeGradientScale.value = (voxel.x + voxel.y + voxel.z) / (6.0 * stepWorld);
        // the local majorant; without a renderer to reduce the volume with, one cell over the
        // whole box carries the global majorant and the walk is the plain Woodcock loop
        const grid = this.majorant.update({
            renderer,
            source: texture,
            packedTF: u.volumeTF.value,
            tfSize: lut.size,
            tfId: lut.texture.id,
            tfVersion: lut.version,
            low,
            high,
            alphaCoef,
            sigmaMax,
        });

        u.volumeMajorant.value = grid.texture;
        u.volumeMajorantCells.value.copy(grid.cells);
        u.volumeMajorantScale.value.copy(grid.scale);

        // the step budget is spent along one ray, so the worst line through the grid is what
        // decides this and not the densest voxel in the box
        const walked = grid.worstLine > 0 ? grid.worstLine : sigmaMax;

        const dense = walked * Math.sqrt(3) > glsl.MAX_STEPS / 2;

        // syncVolume runs on every material edit, so this says it once per crossing of the line
        if (dense && !this.warnedDense) {
            console.warn(`K3D.cinematic: volume is dense enough (sigma ${walked.toFixed(0)}/unit `
                + 'along a ray) that paths may hit the tracking step cap and be absorbed early; '
                + 'lower alpha_coef');
        }

        this.warnedDense = dense;
        u.volumeEnabled.value = 1;

        this.syncDefines(1);
    }

    clearVolume() {
        this.majorant.forget();
        this.uniforms.volumeEnabled.value = 0;
        this.uniforms.volumeTexture.value = null;
        this.syncDefines(0);
    }

    dispose() {
        this.majorant.dispose();
        super.dispose();
    }

    // FEATURE_FOG would otherwise flip one update later and force a second compile of this shader
    syncDefines(volume) {
        this.defines.FEATURE_FOG = this.materials.features.isUsed('FOG') ? 1 : 0;
        this.setDefine('K3D_VOLUME', volume);
    }
}

module.exports = { VolumePathTracingMaterial };
