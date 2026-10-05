const THREE = require('three');
const { GTAOShader, generateMagicSquareNoise } = require('three/examples/jsm/shaders/GTAOShader.js');
const {
    PoissonDenoiseShader, generatePdSamplePointInitializer,
} = require('three/examples/jsm/shaders/PoissonDenoiseShader.js');
const cameraModes = require('../../../core/lib/cameraMode').cameraModes;
const error = require('../../../core/lib/Error').error;
const getSSAAChunkedRender = require('../helpers/SSAAChunkedRender');
const createCinematicPresenter = require('./cinematic/present');

// The upstream denoiser noise (GTAOPass._generateNoise) comes from Math.random and would
// break bit-identical screenshots - a seeded PRNG (mulberry32) replaces it.
function generateDeterministicNoise(size) {
    const data = new Uint8Array(size * size * 4);
    let state = 0x9e3779b9;

    for (let i = 0; i < data.length; i++) {
        state = (state + 0x6d2b79f5) | 0;
        let t = Math.imul(state ^ (state >>> 15), 1 | state);
        t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
        data[i] = ((t ^ (t >>> 14)) >>> 0) % 256;
    }

    const texture = new THREE.DataTexture(data, size, size);

    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.needsUpdate = true;

    return texture;
}

function depthOnBeforeCompile(globalPeelUniforms, shader) {
    if (typeof (shader.defines) === 'undefined') {
        shader.defines = {};
    }

    if (typeof (shader.defines.PROVIDED_FRAG_COORD_Z) === 'undefined') {
        shader.defines.PROVIDED_FRAG_COORD_Z = 0;
    }

    shader.uniforms.uScreenSize = globalPeelUniforms.uScreenSize;
    shader.uniforms.uPrevDepthTexture = globalPeelUniforms.uPrevDepthTexture;
    shader.uniforms.uLayer = globalPeelUniforms.uLayer;
    shader.uniforms.uDepthOffset = globalPeelUniforms.uDepthOffset;

    // Raw depth into a float target: RGBA8 packing quantised at the order of uDepthOffset,
    // turning the classification of close fragments into per-pixel noise. gl_FragCoord.z, not
    // the material's fragCoordZ - the reconstruction disagrees with the colour pass by an ulp.
    shader.fragmentShader = shader.fragmentShader.replace(
        'gl_FragColor = packDepthToRGBA( fragCoordZ );',
        'gl_FragColor = vec4( gl_FragCoord.z, 0.0, 0.0, 1.0 );',
    );

    shader.fragmentShader = require('./shaders/depthShader.fragment.header.glsl') + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace(
        /}(?![\s\S]*})/gm,
        require('./shaders/depthShader.fragment.tail.glsl'),
    );
}

function colorOnBeforeCompile(globalPeelUniforms, shader) {
    if (shader.fragmentShader.indexOf('#include <packing>') === -1) {
        shader.fragmentShader = shader.fragmentShader.replace(
            '#include <common>',
            '#include <common>\n#include <packing>',
        );
    }
    shader.fragmentShader = shader.fragmentShader.replace('#include <packing>', '');
    shader.fragmentShader = `${'#include <packing>\n'
    + 'uniform sampler2D uPrevColorTexture;\n'}${
        shader.fragmentShader}`;

    if (typeof (shader.defines) === 'undefined') {
        shader.defines = {};
    }

    // own depth into attachment 1 - what lets a layer cost one pass instead of two
    shader.defines.K3D_PEEL_DEPTH_OUT = 1;

    depthOnBeforeCompile(globalPeelUniforms, shader);
}

/**
 * AO prepass depth of a cut-out or glowing mesh; a glow writes 3 + glow to .g, which the
 * overlay spares from darkening. Null when the shared override material will do.
 * @param {Object} globalPeelUniforms
 * @param {THREE.Material} material the mesh's colour material, maps already assigned
 * @param {Object} config the mesh's json
 * @return {THREE.MeshDepthMaterial|null}
 */
function createAODepthMaterial(globalPeelUniforms, material, config) {
    const mode = config.alpha_mode || 'opaque';
    const cut = mode !== 'opaque' && Boolean(material.map);
    const emissive = material.emissive
        ? material.emissive.clone().multiplyScalar(material.emissiveIntensity || 0) : null;
    const glow = emissive !== null && (emissive.r + emissive.g + emissive.b) > 0;

    if (!cut && !glow) {
        return null;
    }

    const depth = new THREE.MeshDepthMaterial({
        side: THREE.DoubleSide,
        depthPacking: THREE.RGBADepthPacking,
    });

    if (cut) {
        depth.map = material.map;
        // a blended surface counts as solid from half alpha up, as opacity does for occluders
        depth.alphaTest = mode === 'mask' && typeof (config.alpha_cutoff) !== 'undefined'
            ? config.alpha_cutoff : 0.5;
    }

    const glowMap = glow ? material.emissiveMap : null;

    depth.onBeforeCompile = function (shader) {
        depthOnBeforeCompile(globalPeelUniforms, shader);

        if (!glow) {
            return;
        }

        shader.uniforms.k3dGlow = { value: emissive };

        let glowValue = 'k3dGlow';

        if (glowMap) {
            shader.uniforms.k3dGlowMap = { value: glowMap };
            shader.vertexShader = `varying vec2 vK3dGlowUv;\n${shader.vertexShader.replace(
                '#include <begin_vertex>',
                '#include <begin_vertex>\nvK3dGlowUv = uv;',
            )}`;
            shader.fragmentShader = `uniform sampler2D k3dGlowMap;\nvarying vec2 vK3dGlowUv;\n${
                shader.fragmentShader}`;
            glowValue = 'k3dGlow * texture2D(k3dGlowMap, vK3dGlowUv).rgb';
        }

        shader.fragmentShader = `uniform vec3 k3dGlow;\n${shader.fragmentShader.replace(
            'gl_FragColor = vec4( gl_FragCoord.z, 0.0, 0.0, 1.0 );',
            `vec3 k3dGlowColor = ${glowValue};\n`
            + 'gl_FragColor = vec4( gl_FragCoord.z, '
            + '3.0 + clamp(max(k3dGlowColor.r, max(k3dGlowColor.g, k3dGlowColor.b)), 0.0, 1.0), '
            + '0.0, 1.0 );',
        )}`;
    };
    depth.customProgramCacheKey = () => `k3dAODepth:${glow ? 1 : 0}:${glowMap ? 1 : 0}`;

    return depth;
}

/**
 * Renderer initializer for Three.js library
 * @this K3D.Core world
 * @method Renderer
 * @memberof K3D.Providers.ThreeJS.Initializers
 * @param {Object} K3D current K3D instance
 */
module.exports = function (K3D) {
    const self = this;
    let renderingPromise = null;
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('webgl2', {
        antialias: K3D.parameters.antialias > 0,
        preserveDrawingBuffer: true,
        alpha: true,
        stencil: true,
        powerPreference: 'high-performance',
    });
    const targets = [];
    const mrtTargets = [];
    // An empty layer makes every deeper one empty too, so the loop can stop there. Occlusion
    // queries answer a frame late: the last two layers are probed and the budget shrinks only
    // while both come back empty, which keeps one known-empty layer as headroom. budget < 0 is
    // "not measured"; renders into a target ignore it and peel the full count.
    const peelProbe = {
        budget: -1, peels: -1, pending: null, free: [],
    };
    const compositeScene = new THREE.Scene();
    const planeGeometry = new THREE.PlaneGeometry(2, 2, 1, 1);
    const toneMappingMode = { value: 0 };
    const compositeMaterial = new THREE.ShaderMaterial({
        uniforms: {
            uTextureA: { value: null },
            uTextureB: { value: null },
            uBlit: { value: 0 },
            uToneMapping: toneMappingMode,
            toneMappingExposure: { value: 1.0 },
            tAO: { value: null },
            tAOVol: { value: null },
            tAODepth: { value: null },
            uAoScale: { value: new THREE.Vector2(1, 1) },
            uAoBias: { value: new THREE.Vector2(0, 0) },
            uAoEnabled: { value: 0 },
        },
        vertexShader: require('./shaders/composite.vertex.glsl'),
        fragmentShader: require('./shaders/composite.fragment.glsl'),
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: THREE.CustomBlending,
        blendEquation: THREE.AddEquation,
        blendDst: THREE.OneFactor,
        blendDstAlpha: null,
        blendSrc: THREE.OneMinusDstAlphaFactor,
        blendSrcAlpha: null,
    });
    const globalPeelUniforms = {
        uLayer: { value: 0 },
        uPrevDepthTexture: { value: null },
        uPrevColorTexture: { value: null },
        uScreenSize: { value: new THREE.Vector2(1, 1) },
        // Bridges the ulp disagreement between gl_FragCoord.z of the depth and colour passes -
        // two different programs. The stored depth itself is exact.
        uDepthOffset: { value: 0.0000001 },
    };
    const depthMaterial = new THREE.MeshDepthMaterial();
    // AO prepass variant for wireframes: the wires occlude, the gaps between them do not
    const depthMaterialWireframe = new THREE.MeshDepthMaterial();
    const compositePlane = new THREE.Mesh(planeGeometry, compositeMaterial);
    const cameras = [];

    // --- GTAO (advanced renderer only) ---
    // Full-frame AO computed once per frame/screenshot from a depth prepass (normals are
    // reconstructed from depth - no G-buffer), denoised spatially, then multiplied onto
    // every render of the main scene. Background depth == 1 is discarded by the shaders,
    // so the grid and the backdrop stay untouched.
    const aoTargets = { depth: null, raw: null, denoised: null };
    const fsCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    let aoTexture = null;
    let aoVolTexture = null;
    const aoSize = new THREE.Vector2(1, 1);

    const gtaoMaterial = new THREE.ShaderMaterial({
        defines: { ...GTAOShader.defines, NORMAL_VECTOR_TYPE: 0 },
        uniforms: THREE.UniformsUtils.clone(GTAOShader.uniforms),
        vertexShader: GTAOShader.vertexShader,
        fragmentShader: GTAOShader.fragmentShader,
        depthTest: false,
        depthWrite: false,
    });

    gtaoMaterial.uniforms.tNoise.value = generateMagicSquareNoise();

    const pdMaterial = new THREE.ShaderMaterial({
        defines: {
            ...PoissonDenoiseShader.defines,
            NORMAL_VECTOR_TYPE: 0,
            // generated explicitly: the upstream constructor bakes SAMPLE_VECTORS with
            // exponent 1 and skips regeneration when the field already equals the wish
            SAMPLE_VECTORS: generatePdSamplePointInitializer(16, 2, 2),
        },
        uniforms: THREE.UniformsUtils.clone(PoissonDenoiseShader.uniforms),
        vertexShader: PoissonDenoiseShader.vertexShader,
        fragmentShader: PoissonDenoiseShader.fragmentShader,
        depthTest: false,
        depthWrite: false,
    });

    pdMaterial.uniforms.tNoise.value = generateDeterministicNoise(64);
    pdMaterial.uniforms.lumaPhi.value = 10.0;
    pdMaterial.uniforms.normalPhi.value = 3.0;
    pdMaterial.uniforms.radius.value = 8.0;

    const aoOverlayMaterial = new THREE.ShaderMaterial({
        uniforms: {
            tAO: { value: null },
            tAOVol: { value: null },
            tDepth: { value: null },
            uUvScale: { value: new THREE.Vector2(1, 1) },
            uUvBias: { value: new THREE.Vector2(0, 0) },
        },
        vertexShader: require('./shaders/composite.vertex.glsl'),
        fragmentShader: require('./shaders/aoOverlay.fragment.glsl'),
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: THREE.CustomBlending,
        blendEquation: THREE.AddEquation,
        blendSrc: THREE.ZeroFactor,
        blendDst: THREE.SrcColorFactor,
        blendSrcAlpha: THREE.ZeroFactor,
        blendDstAlpha: THREE.OneFactor,
    });

    // three bakes renderer.toneMapping into programs only for canvas draws, and the
    // whole pipeline composes through targets - the curve is a final blit instead
    let toneTarget = null;
    const toneBlitMaterial = new THREE.ShaderMaterial({
        uniforms: {
            tDiffuse: { value: null },
            uSize: { value: new THREE.Vector2(1, 1) },
            uToneMapping: toneMappingMode,
            uPremultiplied: { value: 1 },
            toneMappingExposure: { value: 1.0 },
        },
        vertexShader: require('./shaders/composite.vertex.glsl'),
        fragmentShader: require('./shaders/toneBlit.fragment.glsl'),
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: THREE.CustomBlending,
        blendEquation: THREE.AddEquation,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.OneMinusSrcAlphaFactor,
    });

    const gtaoScene = new THREE.Scene();
    const pdScene = new THREE.Scene();
    const aoOverlayScene = new THREE.Scene();
    const toneBlitScene = new THREE.Scene();

    [[gtaoScene, gtaoMaterial], [pdScene, pdMaterial], [aoOverlayScene, aoOverlayMaterial],
        [toneBlitScene, toneBlitMaterial]]
        .forEach(([scene, material]) => {
            const plane = new THREE.Mesh(planeGeometry, material);

            plane.frustumCulled = false;
            scene.add(plane);
        });

    // shared by every Volume material: segment bounds for the peel-interleaved
    // composition (#277). Black (z == 0) stands in for the near plane on the first
    // segment; white (z == 1) reads as "no bound" through the peelT sentinel.
    const peelDummyNear = new THREE.DataTexture(new Float32Array([0.0]), 1, 1, THREE.RedFormat, THREE.FloatType);
    const peelDummyFar = new THREE.DataTexture(new Float32Array([1.0]), 1, 1, THREE.RedFormat, THREE.FloatType);

    peelDummyNear.needsUpdate = true;
    peelDummyFar.needsUpdate = true;

    self.k3dVolumePeel = {
        uPeelSegment: { value: 0 },
        uPeelNearTexture: { value: peelDummyNear },
        uPeelFarTexture: { value: peelDummyFar },
        uPeelSize: { value: new THREE.Vector2(1, 1) },
        uPeelInvProjection: { value: new THREE.Matrix4() },
        uPeelInvView: { value: new THREE.Matrix4() },
    };

    self.renderer = new THREE.WebGLRenderer({
        alpha: true,
        precision: 'highp',
        premultipliedAlpha: true,
        antialias: K3D.parameters.antialias > 0,
        logarithmicDepthBuffer: K3D.parameters.logarithmicDepthBuffer,
        canvas,
        context,
    });

    // three r152 turned colour management on and made sRGB the default output. K3D composites its
    // own render targets, so the encode would land only on part of the pipeline - keep it linear.
    self.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;

    if (!context) {
        if (typeof WebGL2RenderingContext !== 'undefined') {
            error(
                'WEBGL Error',
                'Your browser appears to support WebGL2 but it might '
                + 'be disabled. Try updating your OS and/or video card driver.',
                true,
            );
        } else {
            error(
                'WEBGL Error',
                "It's look like your browser has no WebGL2 support.",
                true,
            );
        }
    }

    function handleContextLoss(event) {
        event.preventDefault();
        K3D.disable();
        error('WEBGL Error', 'Context lost.', false);
    }

    K3D.colorOnBeforeCompile = colorOnBeforeCompile.bind(this, globalPeelUniforms);
    K3D.createAODepthMaterial = createAODepthMaterial.bind(this, globalPeelUniforms);

    canvas.addEventListener('webglcontextlost', handleContextLoss, false);

    self.renderer.removeContextLossListener = function () {
        canvas.removeEventListener('webglcontextlost', handleContextLoss);
    };

    const gl = self.renderer.getContext();

    // Absent in fingerprinting-hardened browsers (Tor, privacy.resistFingerprinting). This
    // runs synchronously from the K3D.Core constructor, so it must not be assumed present.
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');

    // kept rather than only logged: a container that falls back to software rendering says so
    // nowhere else, and from Python the console is out of reach
    K3D.glInfo = {
        vendor: gl.getParameter(gl.VENDOR),
        renderer: gl.getParameter(gl.RENDERER),
        unmaskedVendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : null,
        unmaskedRenderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : null,
        version: gl.getParameter(gl.VERSION),
        depthBits: gl.getParameter(gl.DEPTH_BITS),
        stencilBits: gl.getParameter(gl.STENCIL_BITS),
        maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
        maxTextureImageUnits: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS),
    };

    if (debugInfo) {
        console.log('K3D: (UNMASKED_VENDOR_WEBGL)', K3D.glInfo.unmaskedVendor);
        console.log('K3D: (UNMASKED_RENDERER_WEBGL)', K3D.glInfo.unmaskedRenderer);
    }
    console.log('K3D: (depth bits)', K3D.glInfo.depthBits);
    console.log('K3D: (stencil bits)', K3D.glInfo.stencilBits);

    // [0], [1] - layer depth flip/flop (raw z in .r); [2] - accumulator; [3] - layer colour.
    // Half-float accumulation rounds to 8 bits once, at the final blit.
    function ensureTargets(rawWidth, rawHeight) {
        const width = Math.max(1, Math.round(rawWidth));
        const height = Math.max(1, Math.round(rawHeight));

        if (targets.length > 0
            && targets[0].width === width
            && targets[0].height === height) {
            return;
        }

        globalPeelUniforms.uScreenSize.value.set(1 / width, 1 / height);

        while (targets.length) {
            targets.pop().dispose();
        }

        for (let i = 0; i < 2; i++) {
            targets.push(
                new THREE.WebGLRenderTarget(
                    width,
                    height,
                    {
                        minFilter: THREE.NearestFilter,
                        magFilter: THREE.NearestFilter,
                        format: THREE.RedFormat,
                        type: THREE.FloatType,
                    },
                ),
            );
        }

        targets.push(
            new THREE.WebGLRenderTarget(
                width,
                height,
                {
                    minFilter: THREE.NearestFilter,
                    magFilter: THREE.NearestFilter,
                    type: THREE.HalfFloatType,
                    depthBuffer: false,
                },
            ),
        );

        targets.push(
            new THREE.WebGLRenderTarget(
                width,
                height,
                {
                    minFilter: THREE.NearestFilter,
                    magFilter: THREE.NearestFilter,
                    type: THREE.HalfFloatType,
                },
            ),
        );

        while (mrtTargets.length) {
            mrtTargets.pop().dispose();
        }

        peelProbe.budget = -1;

        if (peelProbe.pending !== null) {
            peelProbe.pending.queries.forEach((q) => peelProbe.free.push(q));
            peelProbe.pending = null;
        }
    }

    // single-pass flip/flop: attachment 0 layer colour, attachment 1 the depth the next peel
    // tests against. Allocated on first use.
    function ensureMrtTargets() {
        if (mrtTargets.length > 0) {
            return;
        }

        for (let i = 0; i < 2; i++) {
            const target = new THREE.WebGLRenderTarget(
                targets[0].width,
                targets[0].height,
                {
                    count: 2,
                    minFilter: THREE.NearestFilter,
                    magFilter: THREE.NearestFilter,
                    type: THREE.HalfFloatType,
                },
            );

            target.textures[1].format = THREE.RedFormat;
            target.textures[1].type = THREE.FloatType;

            mrtTargets.push(target);
        }
    }

    function ensureAoTargets(width, height) {
        if (aoTargets.depth !== null
            && aoTargets.depth.width === width
            && aoTargets.depth.height === height) {
            return;
        }

        Object.keys(aoTargets).forEach((key) => {
            if (aoTargets[key] !== null) {
                aoTargets[key].dispose();
            }
        });

        // g carries the volumetric-shell marker (2.0); mesh RGBADepthPacking spills
        // fractional junk < 1.0 there, so the overlay tests g > 1.5
        aoTargets.depth = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            format: THREE.RGFormat,
            type: THREE.FloatType,
        });
        aoTargets.raw = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            type: THREE.HalfFloatType,
            depthBuffer: false,
        });
        aoTargets.denoised = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            type: THREE.HalfFloatType,
            depthBuffer: false,
        });
        // occluder-class separation: volume-shell pixels take AO computed from the
        // shells alone, so meshes do not cast onto the whole ray integral
        aoTargets.depthVol = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            format: THREE.RedFormat,
            type: THREE.FloatType,
        });
        aoTargets.denoisedVol = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            type: THREE.HalfFloatType,
            depthBuffer: false,
        });
    }

    // Full-frame AO for the current camera. Must run before any chunked/strip rendering:
    // the overlay then samples this single buffer, so strips cannot seam.
    function computeAO(width, height) {
        aoTexture = null;

        if (K3D.parameters.renderer !== 'advanced'
            || K3D.parameters.cameraMode === cameraModes.volumeSides) {
            return;
        }

        // GTAO applies pow(ao, aoStrength), so at 0 the buffer is 1 everywhere and the overlay
        // multiplies the frame by itself. The whole chain - a scene depth pass, two fullscreen
        // passes and the overlay - was running to produce that.
        if (K3D.parameters.aoStrength === 0) {
            return;
        }

        const world = K3D.getWorld();
        const box = new THREE.Box3().setFromObject(world.K3DObjects);

        if (box.isEmpty()) {
            return;
        }

        const diagonal = box.getSize(new THREE.Vector3()).length() || 1.0;

        ensureAoTargets(width, height);

        // depth prepass: only real geometry occludes. Lines have no surface for the
        // override material, volumes are back-side boxes that would occlude everything
        // behind them. Impostor spheres carry their own depth material and render in a
        // second, depth-tested pass - the override would rasterise their quads.
        const hidden = [];
        const impostors = [];
        const wireframes = [];
        const occluders = [];

        world.K3DObjects.traverse((obj) => {
            if (!obj.visible) {
                return;
            }
            // a mesh's own depth material does not lift it over the opacity rule below; glass occludes nothing
            if (obj.isMesh && obj.material && !obj.material.isShaderMaterial
                && ((obj.userData.k3dAODepthMaterial && obj.material.opacity < 0.5)
                    || obj.material.transmission > 0)) {
                obj.visible = false;
                hidden.push(obj);
                return;
            }
            if (obj.userData.k3dAODepthMaterial) {
                obj.visible = false;
                impostors.push(obj);
                return;
            }
            // the override material honours neither `wireframe` nor opacity: wireframes get a
            // second pass that keeps the flag, transparent surfaces are dropped
            if (obj.material && obj.material.wireframe && !obj.material.isShaderMaterial) {
                obj.visible = false;
                wireframes.push(obj);
                return;
            }
            if (obj.isPoints || obj.isLine || obj.isSprite
                || (obj.material && (obj.material.isShaderMaterial
                    // opacity >= 0.5 still reads as solid, so it keeps occluding
                    || obj.material.opacity < 0.5))) {
                obj.visible = false;
                hidden.push(obj);

                return;
            }
            if (obj.isMesh) {
                occluders.push(obj);
            }
        });

        // Nothing left that can cast occlusion: the depth prepass would leave the target at its
        // clear value, GTAO and the denoiser would discard every pixel, and the overlay would
        // multiply the frame by 1. A scatter-only or line-only plot paid for all of it.
        if (occluders.length === 0 && impostors.length === 0 && wireframes.length === 0) {
            hidden.forEach((obj) => {
                obj.visible = true;
            });

            return;
        }

        globalPeelUniforms.uLayer.value = 0;

        self.camera.updateMatrixWorld();
        self.renderer.setRenderTarget(aoTargets.depth);
        self.renderer.setClearColor(0xffffff, 1);
        self.renderer.clear(true, true, false);
        self.scene.overrideMaterial = depthMaterial;
        self.renderer.render(self.scene, self.camera);
        self.scene.overrideMaterial = null;

        if (impostors.length > 0) {
            const meshesShown = [];

            world.K3DObjects.traverse((obj) => {
                if (obj.visible && obj.material) {
                    obj.visible = false;
                    meshesShown.push(obj);
                }
            });

            impostors.forEach((obj) => {
                obj.visible = true;
                obj.userData.k3dAOColorMaterial = obj.material;
                obj.material = obj.userData.k3dAODepthMaterial;
            });

            // no clear: depth-tested against the surfaces of the first pass
            self.renderer.render(self.scene, self.camera);

            impostors.forEach((obj) => {
                obj.material = obj.userData.k3dAOColorMaterial;
            });
            meshesShown.forEach((obj) => {
                obj.visible = true;
            });
        }

        if (wireframes.length > 0) {
            const meshesShown = [];

            world.K3DObjects.traverse((obj) => {
                if (obj.visible && obj.material) {
                    obj.visible = false;
                    meshesShown.push(obj);
                }
            });

            wireframes.forEach((obj) => {
                obj.visible = true;
            });

            // wireframe override, no clear: the wires depth-test against the first pass
            self.scene.overrideMaterial = depthMaterialWireframe;
            self.renderer.render(self.scene, self.camera);
            self.scene.overrideMaterial = null;

            wireframes.forEach((obj) => {
                obj.visible = false;
            });
            meshesShown.forEach((obj) => {
                obj.visible = true;
            });
        }

        hidden.concat(impostors, wireframes).forEach((obj) => {
            obj.visible = true;
        });

        const u = gtaoMaterial.uniforms;

        u.tDepth.value = aoTargets.depth.texture;
        u.resolution.value.set(width, height);
        u.cameraNear.value = self.camera.near;
        u.cameraFar.value = self.camera.far;
        u.cameraProjectionMatrix.value.copy(self.camera.projectionMatrix);
        u.cameraProjectionMatrixInverse.value.copy(self.camera.projectionMatrixInverse);
        // view-space units follow the data: aoRadius is a fraction of the scene
        // diagonal, aoStrength the shadow-deepening exponent (plot traits).
        // thickness stays coupled at 2x radius - undercutting the radius makes the
        // horizon test drop samples, which silently disables occlusion in wide cavities
        u.radius.value = K3D.parameters.aoRadius * diagonal;
        u.thickness.value = 2.0 * K3D.parameters.aoRadius * diagonal;
        u.scale.value = K3D.parameters.aoStrength;

        self.renderer.setRenderTarget(aoTargets.raw);
        self.renderer.setClearColor(0xffffff, 1);
        self.renderer.clear(true, false, false);
        self.renderer.render(gtaoScene, fsCamera);

        pdMaterial.uniforms.tDiffuse.value = aoTargets.raw.texture;
        pdMaterial.uniforms.tDepth.value = aoTargets.depth.texture;
        pdMaterial.uniforms.resolution.value.set(width, height);
        pdMaterial.uniforms.cameraProjectionMatrixInverse.value.copy(self.camera.projectionMatrixInverse);
        pdMaterial.uniforms.depthPhi.value = 0.02 * diagonal;

        self.renderer.setRenderTarget(aoTargets.denoised);
        self.renderer.setClearColor(0xffffff, 1);
        self.renderer.clear(true, false, false);
        self.renderer.render(pdScene, fsCamera);

        aoTexture = aoTargets.denoised.texture;
        aoVolTexture = aoTexture;

        const volumeShells = impostors.filter((obj) => obj.userData.k3dVolumeShell);

        if (volumeShells.length > 0) {
            const shown = [];

            world.K3DObjects.traverse((obj) => {
                if (obj.visible && obj.material) {
                    obj.visible = false;
                    shown.push(obj);
                }
            });
            volumeShells.forEach((obj) => {
                obj.visible = true;
                obj.userData.k3dAOColorMaterial = obj.material;
                obj.material = obj.userData.k3dAODepthMaterial;
            });

            self.renderer.setRenderTarget(aoTargets.depthVol);
            self.renderer.setClearColor(0xffffff, 1);
            self.renderer.clear(true, true, false);
            self.renderer.render(self.scene, self.camera);

            volumeShells.forEach((obj) => {
                obj.material = obj.userData.k3dAOColorMaterial;
                obj.visible = false;
            });
            shown.forEach((obj) => {
                obj.visible = true;
            });

            u.tDepth.value = aoTargets.depthVol.texture;

            self.renderer.setRenderTarget(aoTargets.raw);
            self.renderer.setClearColor(0xffffff, 1);
            self.renderer.clear(true, false, false);
            self.renderer.render(gtaoScene, fsCamera);

            pdMaterial.uniforms.tDiffuse.value = aoTargets.raw.texture;
            pdMaterial.uniforms.tDepth.value = aoTargets.depthVol.texture;

            self.renderer.setRenderTarget(aoTargets.denoisedVol);
            self.renderer.setClearColor(0xffffff, 1);
            self.renderer.clear(true, false, false);
            self.renderer.render(pdScene, fsCamera);

            aoVolTexture = aoTargets.denoisedVol.texture;
        }

        self.renderer.setRenderTarget(null);
        aoSize.set(width, height);
    }

    // Multiplies the AO buffer onto whatever the main scene was just rendered into.
    function applyAOOverlay(camera, rt) {
        // only the advanced renderer computes this buffer; cinematic rasterises its preview
        // through the same path and would otherwise keep multiplying the last advanced frame in
        if (aoTexture === null || K3D.parameters.renderer !== 'advanced') {
            return;
        }

        const scale = aoOverlayMaterial.uniforms.uUvScale.value;
        const bias = aoOverlayMaterial.uniforms.uUvBias.value;

        aoOverlayMaterial.uniforms.tAO.value = aoTexture;
        aoOverlayMaterial.uniforms.tAOVol.value = aoVolTexture;
        aoOverlayMaterial.uniforms.tDepth.value = aoTargets.depth.texture;

        if (rt && camera.view && camera.view.enabled) {
            // strip target: gl_FragCoord is target-local, the frustum covers
            // camera.view rows of the full frame (stretched over the whole target)
            const v = camera.view;

            scale.set(
                v.width / (rt.width * v.fullWidth),
                v.height / (rt.height * v.fullHeight),
            );
            bias.set(v.offsetX / v.fullWidth, (v.fullHeight - v.offsetY - v.height) / v.fullHeight);
        } else if (rt) {
            scale.set(1 / rt.width, 1 / rt.height);
            bias.set(0, 0);
        } else {
            // canvas: gl_FragCoord is global and counted in drawing-buffer pixels, which is not
            // getSize() once setPixelRatio is anything but 1 - minimum_fps moves it every frame
            const buffer = new THREE.Vector2();

            self.renderer.getDrawingBufferSize(buffer);
            scale.set(1 / buffer.x, 1 / buffer.y);
            bias.set(0, 0);
        }

        self.renderer.render(aoOverlayScene, fsCamera);
    }

    // An unpatched material blends (WebGL2 shares blend state across attachments) and never
    // writes attachment 1, so one of them puts the whole frame back on two passes. Volumes are
    // exempt - they are hidden for the geometry passes.
    function scenePeelsWithMrt() {
        let supported = true;

        K3D.getWorld().K3DObjects.traverse((obj) => {
            if (!supported || !obj.visible || !obj.material || obj.userData.k3dVolumeSegments) {
                return;
            }

            if (Array.isArray(obj.material) || obj.material.userData.k3dPeelDepthOut !== true) {
                supported = false;
            }
        });

        return supported;
    }

    function readPeelProbe() {
        const pending = peelProbe.pending;

        if (pending === null) {
            return;
        }

        if (!pending.queries.every((q) => gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE))) {
            return;
        }

        const drawn = pending.queries.map((q) => gl.getQueryParameter(q, gl.QUERY_RESULT) > 0);

        peelProbe.pending = null;
        pending.queries.forEach((q) => peelProbe.free.push(q));

        if (pending.peels !== K3D.parameters.depthPeels) {
            return;
        }

        if (drawn[drawn.length - 1]) {
            peelProbe.budget = Math.min(pending.peels, pending.layers + 1);
        } else if (drawn.length > 1 && !drawn[0]) {
            peelProbe.budget = Math.max(0, pending.layers - 1);
        } else {
            peelProbe.budget = pending.layers;
        }
    }

    // The peeling itself happens in depthShader.fragment.tail: each pass discards fragments
    // not strictly deeper than the previous layer.
    const peelViewport = new THREE.Vector4();
    const peelCanvasSize = new THREE.Vector2();
    const mrtClearColor = [0, 0, 0, 0];
    const mrtClearDepth = [1, 0, 0, 1];

    function depthPeelRender(scene, camera, rt) {
        let fullFrame = false;

        if (typeof (rt) === 'undefined') {
            rt = null;
            // peels cover exactly the region the composite lands in; with renderingSteps > 1 that
            // is one strip, and a full-frame target would hold it stretched and get point-sampled
            self.renderer.getViewport(peelViewport);
            ensureTargets(peelViewport.z, peelViewport.w);

            self.renderer.getSize(peelCanvasSize);

            fullFrame = peelViewport.x === 0 && peelViewport.y === 0
                && peelViewport.z === peelCanvasSize.x
                && peelViewport.w === peelCanvasSize.y;
        } else {
            ensureTargets(rt.width, rt.height);
        }

        readPeelProbe();

        // restore exactly what was hidden - a filter-based restore would resurrect
        // objects the user hid while their opacity was 0
        const opacityHidden = [];

        K3D.getWorld().K3DObjects.children.forEach((obj) => {
            if (obj.visible && obj.material && obj.material.opacity <= 0.0) {
                obj.visible = false;
                opacityHidden.push(obj);
            }
        });

        globalPeelUniforms.uLayer.value = 0;
        globalPeelUniforms.uPrevDepthTexture.value = null;

        // AO multiplies each layer and segment during composition, not the final
        // blit - the finished image mixes volume light with the geometry behind it
        compositeMaterial.uniforms.uAoEnabled.value = aoTexture !== null ? 1 : 0;

        if (aoTexture !== null) {
            compositeMaterial.uniforms.tAO.value = aoTexture;
            compositeMaterial.uniforms.tAOVol.value = aoVolTexture;
            compositeMaterial.uniforms.tAODepth.value = aoTargets.depth.texture;

            if (camera.view && camera.view.enabled) {
                // strip target: vUv covers camera.view rows of the full-frame AO buffer
                const v = camera.view;

                compositeMaterial.uniforms.uAoScale.value.set(v.width / v.fullWidth, v.height / v.fullHeight);
                compositeMaterial.uniforms.uAoBias.value.set(
                    v.offsetX / v.fullWidth,
                    (v.fullHeight - v.offsetY - v.height) / v.fullHeight,
                );
            } else {
                compositeMaterial.uniforms.uAoScale.value.set(1, 1);
                compositeMaterial.uniforms.uAoBias.value.set(0, 0);
            }
        }

        compositeMaterial.uniforms.uBlit.value = 1;
        compositeMaterial.blendSrc = THREE.OneMinusDstAlphaFactor;
        compositeMaterial.blendDst = THREE.OneFactor;

        gl.colorMask(true, true, true, true);
        gl.depthMask(true);

        // accumulator
        self.renderer.setRenderTarget(targets[2]);
        self.renderer.setClearColor(0, 0);
        self.renderer.clear();

        const peels = K3D.parameters.depthPeels;

        // the budget converges for one peel count, and the probe only walks it by one layer per
        // frame: kept across a change, raising depth_peels takes a dozen frames to take effect
        if (peelProbe.peels !== peels) {
            peelProbe.peels = peels;
            peelProbe.budget = -1;
        }

        // the budget needs the whole frame in one call: a screenshot has to be exact, and a strip
        // or a volumeSides quadrant would impose its own depth complexity on the rest of the frame
        const layers = (fullFrame && peelProbe.budget >= 0)
            ? Math.min(peels, peelProbe.budget)
            : peels;
        const probing = fullFrame && peelProbe.pending === null;
        const probeQueries = [];
        const useMrt = scenePeelsWithMrt();

        if (useMrt) {
            ensureMrtTargets();
        }

        function renderSceneProbed(index) {
            const probed = probing && index >= layers - 1;
            let query = null;

            if (probed) {
                query = peelProbe.free.pop() || gl.createQuery();
                gl.beginQuery(gl.ANY_SAMPLES_PASSED_CONSERVATIVE, query);
            }

            self.renderer.render(scene, camera);

            if (probed) {
                gl.endQuery(gl.ANY_SAMPLES_PASSED_CONSERVATIVE);
                probeQueries.push(query);
            }
        }

        function renderLayerColor(index) {
            self.renderer.setRenderTarget(targets[3]);
            self.renderer.setClearColor(0, 0);
            self.renderer.clear(true, true, false);
            renderSceneProbed(index);
        }

        function renderLayerDepth(target) {
            self.renderer.setRenderTarget(target);
            self.renderer.setClearColor(0xffffff, 1);
            self.renderer.clear(true, true, false);

            scene.overrideMaterial = depthMaterial;
            self.renderer.render(scene, camera);
            scene.overrideMaterial = null;
        }

        // colour and depth in one pass; the attachments need different clear values (empty
        // layer, far plane), which a single clear colour cannot express
        function renderLayerMrt(index) {
            self.renderer.setRenderTarget(mrtTargets[index % 2]);
            gl.clearBufferfv(gl.COLOR, 0, mrtClearColor);
            gl.clearBufferfv(gl.COLOR, 1, mrtClearDepth);
            self.renderer.clear(false, true, false);
            renderSceneProbed(index);
        }

        function layerDepthTexture(index) {
            return useMrt ? mrtTargets[index % 2].textures[1] : targets[index % 2].texture;
        }

        function compositeTexture(texture) {
            compositeMaterial.uniforms.uTextureA.value = texture;
            self.renderer.setRenderTarget(targets[2]);
            self.renderer.render(compositeScene, camera);
        }

        // volumes leave the geometry passes: their box neither peels nor occludes,
        // and the march runs as per-segment passes interleaved between the layers (#277)
        const volumeObjects = [];

        K3D.getWorld().K3DObjects.traverse((obj) => {
            if (obj.visible && obj.userData.k3dVolumeSegments) {
                volumeObjects.push(obj);
                obj.visible = false;
            }
        });

        function renderVolumeSegments(nearTexture, farTexture) {
            if (volumeObjects.length === 0) {
                return;
            }

            const u = self.k3dVolumePeel;
            const shown = [];

            u.uPeelSegment.value = 1;
            u.uPeelNearTexture.value = nearTexture;
            u.uPeelFarTexture.value = farTexture;
            u.uPeelSize.value.set(1.0 / targets[0].width, 1.0 / targets[0].height);
            u.uPeelInvProjection.value.copy(camera.projectionMatrixInverse);
            u.uPeelInvView.value.copy(camera.matrixWorld);

            // leaves only - hiding the K3DObjects group itself would hide the volumes too
            K3D.getWorld().K3DObjects.traverse((obj) => {
                if (obj.visible && obj.material) {
                    obj.visible = false;
                    shown.push(obj);
                }
            });
            volumeObjects.forEach((obj) => {
                obj.visible = true;
            });

            self.renderer.setRenderTarget(targets[3]);
            self.renderer.setClearColor(0, 0);
            self.renderer.clear(true, true, false);
            self.renderer.render(scene, camera);

            volumeObjects.forEach((obj) => {
                obj.visible = false;
            });
            shown.forEach((obj) => {
                obj.visible = true;
            });

            // the march output is already premultiplied - composite it as-is
            compositeMaterial.uniforms.uBlit.value = 2;
            compositeTexture(targets[3].texture);
            compositeMaterial.uniforms.uBlit.value = 1;

            u.uPeelSegment.value = 0;
        }

        function renderLayer(index) {
            if (useMrt) {
                renderLayerMrt(index);

                return;
            }

            // only the volume segments read the deepest layer's depth
            if (index === layers && volumeObjects.length === 0) {
                return;
            }

            renderLayerDepth(targets[index % 2]);
        }

        function finishLayer(index) {
            if (!useMrt) {
                renderLayerColor(index);
            }

            compositeTexture(useMrt ? mrtTargets[index % 2].textures[0] : targets[3].texture);
        }

        camera.updateMatrixWorld();

        // layer 0: uLayer == 0, so the tail discards nothing
        renderLayer(0);
        renderVolumeSegments(peelDummyNear, layerDepthTexture(0));
        finishLayer(0);

        for (let i = 0; i < layers; i++) {
            globalPeelUniforms.uPrevDepthTexture.value = layerDepthTexture(i);
            globalPeelUniforms.uLayer.value = i + 1;

            renderLayer(i + 1);
            renderVolumeSegments(layerDepthTexture(i), layerDepthTexture(i + 1));
            finishLayer(i + 1);
        }

        globalPeelUniforms.uLayer.value = 0;
        renderVolumeSegments(layerDepthTexture(layers), peelDummyFar);

        volumeObjects.forEach((obj) => {
            obj.visible = true;
        });

        if (probeQueries.length > 0) {
            peelProbe.pending = { queries: probeQueries, layers, peels };
        }

        // final blit of the accumulator
        globalPeelUniforms.uLayer.value = 0;

        self.renderer.setRenderTarget(rt);

        compositeMaterial.uniforms.uBlit.value = 0;
        compositeMaterial.blendSrc = THREE.OneFactor;
        compositeMaterial.blendDst = THREE.OneMinusSrcAlphaFactor;
        compositeMaterial.blendSrcAlpha = null;
        compositeMaterial.blendDstAlpha = null;
        compositeMaterial.uniforms.uTextureA.value = targets[2].texture;

        self.renderer.render(compositeScene, camera);

        opacityHidden.forEach((obj) => {
            obj.visible = true;
        });
    }

    function directRender(scene, camera, rt) {
        if (typeof (rt) === 'undefined') {
            rt = null;
        }

        // the tone curve needs the frame in a texture first - render via an
        // intermediate target mirroring the destination, then blit through the curve
        if (toneMappingMode.value !== 0 && scene === self.scene) {
            const width = rt ? rt.width : K3D.getWorld().width;
            const height = rt ? rt.height : K3D.getWorld().height;
            const viewport = new THREE.Vector4();

            self.renderer.getViewport(viewport);

            if (toneTarget === null || toneTarget.width !== width || toneTarget.height !== height) {
                if (toneTarget !== null) {
                    toneTarget.dispose();
                }

                toneTarget = new THREE.WebGLRenderTarget(width, height, {
                    minFilter: THREE.NearestFilter,
                    magFilter: THREE.NearestFilter,
                    type: THREE.HalfFloatType,
                });
            }

            self.renderer.setRenderTarget(toneTarget);
            self.renderer.setViewport(viewport);
            self.renderer.setClearColor(0, 0);
            self.renderer.clear();
            self.renderer.render(scene, camera);

            // AO on the linear image, before the curve
            applyAOOverlay(camera, rt);

            toneBlitMaterial.uniforms.tDiffuse.value = toneTarget.texture;
            toneBlitMaterial.uniforms.uSize.value.set(width, height);
            toneBlitMaterial.uniforms.uPremultiplied.value = 1;

            self.renderer.setRenderTarget(rt);
            self.renderer.setViewport(viewport);
            self.renderer.render(toneBlitScene, fsCamera);

            return;
        }

        self.renderer.setRenderTarget(rt);
        self.renderer.render(scene, camera);

        // grid and axes go through directRender too - only the main scene gets AO
        if (scene === self.scene) {
            applyAOOverlay(camera, rt);
        }
    }

    // everything cinematic past the choice of mode: presentation, compose, OIDN, screenshots
    const cinematicPresenter = createCinematicPresenter(K3D, self, {
        fsCamera,
        planeGeometry,
        depthMaterial,
        globalPeelUniforms,
        peelDummyNear,
        toneBlitMaterial,
        toneBlitScene,
        directRender,
    });

    function render() {
        if (K3D.parameters.renderer === 'cinematic') {
            return cinematicPresenter.render();
        }

        // leaving cinematic: an in-flight accumulation would paint over the raster frames
        cinematicPresenter.leave();

        if (cameras.length === 0) {
            for (let i = 0; i < 3; i++) {
                cameras.push(self.camera.clone());
            }
        }

        return new Promise((resolve) => {
            if (K3D.disabling) {
                resolve(null);
                return;
            }

            const size = new THREE.Vector2();

            self.renderer.getSize(size);

            K3D.refreshGrid();

            self.renderer.clippingPlanes = [];

            self.camera.updateMatrixWorld();

            self.renderer.clear();

            self.renderer.setViewport(0, 0, size.x, size.y);
            self.renderer.render(self.gridScene, self.camera);

            K3D.parameters.clippingPlanes.forEach((plane) => {
                self.renderer.clippingPlanes.push(new THREE.Plane(new THREE.Vector3().fromArray(plane), plane[3]));
            });

            K3D.dispatch(K3D.events.BEFORE_RENDER);

            computeAO(size.x, size.y);

            const currentRenderMethod = K3D.parameters.depthPeels > 0 ? depthPeelRender : directRender;

            let p = Promise.resolve();
            const originalControlsEnabledState = self.controls.enabled;

            function renderPass(x, y, width, height, viewport) {
                const chunkWidths = [];

                if (K3D.parameters.renderingSteps > 1) {
                    const s = width / K3D.parameters.renderingSteps;

                    for (let i = 0; i < K3D.parameters.renderingSteps; i++) {
                        const o1 = Math.round(i * s);
                        const o2 = Math.min(Math.round((i + 1) * s), width);
                        chunkWidths.push([o1, o2 - o1]);
                    }
                }

                if (K3D.parameters.renderingSteps > 1) {
                    self.controls.enabled = false;

                    if (self.controls.beforeRender) {
                        p = p.then(() => {
                            self.controls.beforeRender(viewport);

                            if (viewport < 3) {
                                cameras[viewport].copy(self.controls.object, false);
                            }
                        });
                    }

                    chunkWidths.forEach((c) => {
                        p = p.then(() => {
                            // the offset belongs on the rendered camera and subdivides this pass,
                            // which in volumeSides mode is one quadrant
                            const chunkCamera = viewport < 3 ? cameras[viewport] : self.camera;

                            self.renderer.setViewport(x + c[0], y, c[1], height);
                            chunkCamera.setViewOffset(width, height, c[0], 0, c[1], height);

                            currentRenderMethod(self.scene, chunkCamera);
                        });

                        // one macrotask instead of a fixed 50 ms per chunk
                        p = p.then(() => new Promise((chunkResolve) => {
                            setTimeout(chunkResolve, 0);
                        }));
                    });

                    if (self.controls.afterRender) {
                        p = p.then(() => {
                            self.controls.afterRender(viewport);
                        });
                    }
                } else {
                    p = p.then(() => {
                        if (self.controls.beforeRender) {
                            self.controls.beforeRender(viewport);

                            if (viewport < 3) {
                                cameras[viewport].copy(self.controls.object, false);
                            }
                        }

                        self.renderer.setViewport(x, y, width, height);

                        if (viewport < 3) {
                            currentRenderMethod(self.scene, cameras[viewport]);
                        } else {
                            currentRenderMethod(self.scene, self.camera);
                        }

                        if (self.controls.afterRender) {
                            self.controls.afterRender(viewport);
                        }
                    });
                }
            }

            if (K3D.parameters.cameraMode === cameraModes.volumeSides) {
                renderPass(0, size.y / 2, size.x / 2, size.y / 2, 0);
                renderPass(0, 0, size.x / 2, size.y / 2, 1);
                renderPass(size.x / 2, size.y / 2, size.x / 2, size.y / 2, 2);
                renderPass(size.x / 2, 0, size.x / 2, size.y / 2, 3);
            } else {
                renderPass(0, 0, size.x, size.y);
            }

            p = p.then(() => {
                self.controls.enabled = originalControlsEnabledState;

                self.renderer.setViewport(
                    size.x - self.axesHelper.width,
                    0,
                    self.axesHelper.width,
                    self.axesHelper.height,
                );
                // the compass is an overlay, not part of the scene: global clipping planes are
                // still set here and would cut it, which the offscreen path already avoids
                const scenePlanes = self.renderer.clippingPlanes;

                self.renderer.clippingPlanes = [];
                self.renderer.render(self.axesHelper.scene, self.axesHelper.camera);
                self.renderer.clippingPlanes = scenePlanes;

                self.renderer.setViewport(0, 0, size.x, size.y);
                self.camera.clearViewOffset();
                cameras.forEach((camera) => camera.clearViewOffset());

                K3D.dispatch(K3D.events.RENDERED);

                resolve(true);
            });
        });
    }

    compositePlane.frustumCulled = false;
    compositeScene.add(compositePlane);

    depthMaterial.side = THREE.DoubleSide;
    depthMaterial.depthPacking = THREE.RGBADepthPacking;
    depthMaterial.onBeforeCompile = depthOnBeforeCompile.bind(null, globalPeelUniforms);
    depthMaterial.needsUpdate = true;

    depthMaterialWireframe.side = THREE.DoubleSide;
    depthMaterialWireframe.depthPacking = THREE.RGBADepthPacking;
    depthMaterialWireframe.wireframe = true;
    depthMaterialWireframe.onBeforeCompile = depthOnBeforeCompile.bind(null, globalPeelUniforms);
    depthMaterialWireframe.needsUpdate = true;

    this.renderer.setClearColor(0, 0);
    this.renderer.autoClear = false;

    // NOT renderer.toneMapping: three bakes it into programs only for canvas draws
    // (getParameters: currentRenderTarget === null), and screenshots, strips and the
    // peel pipeline all compose through targets. One uniform, zero recompiles.
    self.applyToneMapping = function (name) {
        const map = {
            none: 0,
            agx: 1,
            aces: 2,
        };

        toneMappingMode.value = map[name] || 0;
    };

    // in cinematic the unforced render requests of one tick collapse into a single
    // accumulation; headless drops them entirely - there every frame is forced
    let coalescedRender = null;

    this.render = function (force) {
        K3D.labels = [];

        if (K3D.parameters.renderer === 'cinematic' && !force) {
            if (typeof window !== 'undefined' && typeof window.headlessK3D !== 'undefined') {
                return Promise.resolve(null);
            }

            // abort now, not when the coalesced render starts: the in-flight accumulation is superseded
            cinematicPresenter.abort();

            if (coalescedRender === null) {
                coalescedRender = new Promise((resolve) => {
                    setTimeout(() => {
                        coalescedRender = null;
                        Promise.resolve(self.render(true)).then(resolve);
                    }, 0);
                });
            }

            return coalescedRender;
        }

        // a forced render queues behind an accumulation whose result is already obsolete - abandon it
        if (force && K3D.parameters.renderer === 'cinematic') {
            cinematicPresenter.abort();
        }

        // an unforced request arriving while a render is in flight is dropped - the caller gets
        // the frame already on its way. A forced one queues behind it instead, never in parallel.
        if (renderingPromise === null) {
            // clear the queue only while this link is still its tail: clearing it from an older
            // link lets the next unforced request start a second render beside the one in flight
            const link = render().then(() => {
                if (renderingPromise === link) {
                    renderingPromise = null;
                }
            });

            renderingPromise = link;

            return renderingPromise;
        }

        if (force) {
            const link = renderingPromise.then(render).then(() => {
                if (renderingPromise === link) {
                    renderingPromise = null;
                }
            });

            renderingPromise = link;
        }

        return renderingPromise;
    };

    this.renderOffScreen = function (width, height) {
        if (K3D.parameters.renderer === 'cinematic') {
            return cinematicPresenter.offScreen(width, height);
        }

        const chunkHeights = [];
        const chunkCount = Math.max(Math.min(128, K3D.parameters.renderingSteps), 1);
        const aaLevel = Math.max(Math.min(5, K3D.parameters.antialias), 0);
        const currentRenderMethod = K3D.parameters.depthPeels > 0 ? depthPeelRender : directRender;

        const s = height / chunkCount;

        const size = new THREE.Vector2();

        self.renderer.getSize(size);

        const scale = Math.max(width / size.x, height / size.y);

        for (let i = 0; i < chunkCount; i++) {
            const o1 = Math.round(i * s);
            const o2 = Math.min(Math.round((i + 1) * s), height);
            chunkHeights.push([o1, o2 - o1]);
        }

        // Full height even when chunked: the chunks are selected by scissor, so they share one
        // target and one projection. It costs no more than this path already paid - the grid
        // pass allocated a full-height target of its own - and it costs less, because that
        // second allocation is now the same object.
        const rt = new THREE.WebGLRenderTarget(width, height, {
            type: THREE.FloatType,
        });

        const rtAxesHelper = new THREE.WebGLRenderTarget(
            self.axesHelper.width * scale,
            self.axesHelper.height * scale,
            {
                type: THREE.FloatType,
            },
        );
        self.renderer.clippingPlanes = [];

        return getSSAAChunkedRender(
            self.renderer,
            self.axesHelper.scene,
            self.axesHelper.camera,
            rtAxesHelper,
            rtAxesHelper.width,
            rtAxesHelper.height,
            [[0, rtAxesHelper.height]],
            aaLevel,
            directRender,
        ).then((result) => {
            const axesHelper = new Uint8ClampedArray(width * height * 4);

            for (let y = 0; y < rtAxesHelper.height; y++) {
                // fast row-copy
                axesHelper.set(
                    result.slice(y * rtAxesHelper.width * 4, (y + 1) * rtAxesHelper.width * 4),
                    (y * width + width - rtAxesHelper.width) * 4,
                );
            }

            // the grid is read out before the scene draws over it

            return getSSAAChunkedRender(
                self.renderer,
                self.gridScene,
                self.camera,
                rt,
                width,
                height,
                [[0, height]],
                aaLevel,
                directRender,
            ).then((grid) => {
                K3D.parameters.clippingPlanes.forEach((plane) => {
                    self.renderer.clippingPlanes.push(new THREE.Plane(new THREE.Vector3().fromArray(plane), plane[3]));
                });

                computeAO(width, height);

                return getSSAAChunkedRender(
                    self.renderer,
                    self.scene,
                    self.camera,
                    rt,
                    width,
                    height,
                    chunkHeights,
                    aaLevel,
                    currentRenderMethod,
                ).then((scene) => {
                    rt.dispose();
                    rtAxesHelper.dispose();
                    return [grid, scene, axesHelper];
                });
            });
        });
    };
};
