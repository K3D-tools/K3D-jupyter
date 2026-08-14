const THREE = require('three');
const { GTAOShader, generateMagicSquareNoise } = require('three/examples/jsm/shaders/GTAOShader.js');
const { PoissonDenoiseShader, generatePdSamplePointInitializer } = require('three/examples/jsm/shaders/PoissonDenoiseShader.js');
const cameraModes = require('../../../core/lib/cameraMode').cameraModes;
const error = require('../../../core/lib/Error').error;
const getSSAAChunkedRender = require('../helpers/SSAAChunkedRender');

// The upstream denoiser noise (GTAOPass._generateNoise) comes from Math.random and would
// break bit-identical screenshots - a seeded PRNG (mulberry32) replaces it.
function generateDeterministicNoise(size) {
    const data = new Uint8Array(size * size * 4);
    let state = 0x9e3779b9;

    for (let i = 0; i < data.length; i++) {
        state = (state + 0x6d2b79f5) | 0;
        let t = Math.imul(state ^ (state >>> 15), 1 | state);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        data[i] = ((t ^ (t >>> 14)) >>> 0) % 256;
    }

    const texture = new THREE.DataTexture(data, size, size);

    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.needsUpdate = true;

    return texture;
}

function depthOnBeforeCompile(globalPeelUniforms, shader) {
    if (typeof (shader.defines) == 'undefined') {
        shader.defines = {};
    }

    if (typeof (shader.defines.PROVIDED_FRAG_COORD_Z) == 'undefined') {
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
    shader.fragmentShader = shader.fragmentShader.replace(/}(?![\s\S]*})/gm, require('./shaders/depthShader.fragment.tail.glsl'));
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

    depthOnBeforeCompile(globalPeelUniforms, shader);
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
    let aoSize = new THREE.Vector2(1, 1);

    const gtaoMaterial = new THREE.ShaderMaterial({
        defines: Object.assign({}, GTAOShader.defines, {
            NORMAL_VECTOR_TYPE: 0,
        }),
        uniforms: THREE.UniformsUtils.clone(GTAOShader.uniforms),
        vertexShader: GTAOShader.vertexShader,
        fragmentShader: GTAOShader.fragmentShader,
        depthTest: false,
        depthWrite: false,
    });

    gtaoMaterial.uniforms.tNoise.value = generateMagicSquareNoise();

    const pdMaterial = new THREE.ShaderMaterial({
        defines: Object.assign({}, PoissonDenoiseShader.defines, {
            NORMAL_VECTOR_TYPE: 0,
            // generated explicitly: the upstream constructor bakes SAMPLE_VECTORS with
            // exponent 1 and skips regeneration when the field already equals the wish
            SAMPLE_VECTORS: generatePdSamplePointInitializer(16, 2, 2),
        }),
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

    canvas.addEventListener('webglcontextlost', handleContextLoss, false);

    self.renderer.removeContextLossListener = function () {
        canvas.removeEventListener('webglcontextlost', handleContextLoss);
    };

    const gl = self.renderer.getContext();

    // Absent in fingerprinting-hardened browsers (Tor, privacy.resistFingerprinting). This
    // runs synchronously from the K3D.Core constructor, so it must not be assumed present.
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');

    if (debugInfo) {
        console.log('K3D: (UNMASKED_VENDOR_WEBGL)', gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL));
        console.log('K3D: (UNMASKED_RENDERER_WEBGL)', gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL));
    }
    console.log('K3D: (depth bits)', gl.getParameter(gl.DEPTH_BITS));
    console.log('K3D: (stencil bits)', gl.getParameter(gl.STENCIL_BITS));

    // [0], [1] - layer depth flip/flop (raw z in .r); [2] - accumulator; [3] - layer colour.
    // Half-float accumulation rounds to 8 bits once, at the final blit.
    function ensureTargets(width, height) {
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

        aoTargets.depth = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            format: THREE.RedFormat,
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
    }

    // Full-frame AO for the current camera. Must run before any chunked/strip rendering:
    // the overlay then samples this single buffer, so strips cannot seam.
    function computeAO(width, height) {
        aoTexture = null;

        if (K3D.parameters.renderer !== 'advanced'
            || K3D.parameters.cameraMode === cameraModes.volumeSides) {
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

        world.K3DObjects.traverse((obj) => {
            if (!obj.visible) {
                return;
            }
            if (obj.userData.k3dAODepthMaterial) {
                obj.visible = false;
                impostors.push(obj);
                return;
            }
            if (obj.isPoints || obj.isLine || obj.isSprite
                || (obj.material && obj.material.isShaderMaterial)) {
                obj.visible = false;
                hidden.push(obj);
            }
        });

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

        hidden.concat(impostors).forEach((obj) => {
            obj.visible = true;
        });

        const u = gtaoMaterial.uniforms;

        u.tDepth.value = aoTargets.depth.texture;
        u.resolution.value.set(width, height);
        u.cameraNear.value = self.camera.near;
        u.cameraFar.value = self.camera.far;
        u.cameraProjectionMatrix.value.copy(self.camera.projectionMatrix);
        u.cameraProjectionMatrixInverse.value.copy(self.camera.projectionMatrixInverse);
        // view-space units follow the data, not any fixed scale; the exponent deepens
        // the shadows (author's call: the default footprint read too shallow).
        // thickness must not undercut radius - the horizon test drops samples with
        // a depth delta beyond it, which silently disables occlusion in wide cavities.
        u.radius.value = 0.07 * diagonal;
        u.thickness.value = 0.14 * diagonal;
        u.scale.value = 1.8;

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

        self.renderer.setRenderTarget(null);

        aoTexture = aoTargets.denoised.texture;
        aoSize.set(width, height);
    }

    // Multiplies the AO buffer onto whatever the main scene was just rendered into.
    function applyAOOverlay(camera, rt) {
        if (aoTexture === null) {
            return;
        }

        const scale = aoOverlayMaterial.uniforms.uUvScale.value;
        const bias = aoOverlayMaterial.uniforms.uUvBias.value;

        aoOverlayMaterial.uniforms.tAO.value = aoTexture;

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
            // canvas: gl_FragCoord is global, the AO buffer covers the whole canvas
            scale.set(1 / aoSize.x, 1 / aoSize.y);
            bias.set(0, 0);
        }

        self.renderer.render(aoOverlayScene, fsCamera);
    }

    // The peeling itself happens in depthShader.fragment.tail: each pass discards fragments
    // not strictly deeper than the previous layer.
    function depthPeelRender(scene, camera, rt) {
        if (typeof (rt) === 'undefined') {
            rt = null;
            ensureTargets(K3D.getWorld().width, K3D.getWorld().height);
        } else {
            ensureTargets(rt.width, rt.height);
        }

        K3D.getWorld().K3DObjects.children.forEach((obj) => {
            if (obj.material && obj.material.opacity <= 0.0) {
                obj.visible = false;
            }
        });

        globalPeelUniforms.uLayer.value = 0;
        globalPeelUniforms.uPrevDepthTexture.value = null;

        compositeMaterial.uniforms.uBlit.value = 1;
        compositeMaterial.blendSrc = THREE.OneMinusDstAlphaFactor;
        compositeMaterial.blendDst = THREE.OneFactor;

        gl.colorMask(true, true, true, true);
        gl.depthMask(true);

        // accumulator
        self.renderer.setRenderTarget(targets[2]);
        self.renderer.setClearColor(0, 0);
        self.renderer.clear();

        function renderLayerColor() {
            self.renderer.setRenderTarget(targets[3]);
            self.renderer.setClearColor(0, 0);
            self.renderer.clear(true, true, false);
            self.renderer.render(scene, camera);
        }

        function renderLayerDepth(target) {
            self.renderer.setRenderTarget(target);
            self.renderer.setClearColor(0xffffff, 1);
            self.renderer.clear(true, true, false);

            scene.overrideMaterial = depthMaterial;
            self.renderer.render(scene, camera);
            scene.overrideMaterial = null;
        }

        function compositeLayer() {
            compositeMaterial.uniforms.uTextureA.value = targets[3].texture;
            self.renderer.setRenderTarget(targets[2]);
            self.renderer.render(compositeScene, camera);
        }

        // layer 0: uLayer == 0, so the tail discards nothing
        renderLayerDepth(targets[0]);
        renderLayerColor();
        compositeLayer();

        for (let i = 0; i < K3D.parameters.depthPeels; i++) {
            globalPeelUniforms.uPrevDepthTexture.value = targets[i % 2].texture;
            globalPeelUniforms.uLayer.value = i + 1;

            renderLayerColor();
            compositeLayer();
            renderLayerDepth(targets[(i + 1) % 2]);
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

        applyAOOverlay(camera, rt);

        K3D.getWorld().K3DObjects.children.forEach((obj) => {
            if (obj.material && obj.material.opacity <= 0.0) {
                obj.visible = true;
            }
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

    function render() {
        const currentRenderMethod = K3D.parameters.depthPeels > 0 ? depthPeelRender : directRender;

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
                            self.renderer.setViewport(x + c[0], y, c[1], height);
                            self.camera.setViewOffset(size.x, size.y, c[0], 0, c[1], size.y);

                            if (viewport < 3) {
                                currentRenderMethod(self.scene, cameras[viewport]);
                            } else {
                                currentRenderMethod(self.scene, self.camera);
                            }
                        });

                        p = p.then(() => new Promise((chunkResolve) => {
                            setTimeout(chunkResolve, 50);
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
                self.renderer.render(self.axesHelper.scene, self.axesHelper.camera);

                self.renderer.setViewport(0, 0, size.x, size.y);
                self.camera.clearViewOffset();

                K3D.dispatch(K3D.events.RENDERED);

                resolve(true);

                if (K3D.autoRendering) {
                    requestAnimationFrame(render);
                }
            });
        });
    }

    compositePlane.frustumCulled = false;
    compositeScene.add(compositePlane);

    depthMaterial.side = THREE.DoubleSide;
    depthMaterial.depthPacking = THREE.RGBADepthPacking;
    depthMaterial.onBeforeCompile = depthOnBeforeCompile.bind(null, globalPeelUniforms);
    depthMaterial.needsUpdate = true;

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

    this.render = function (force) {
        K3D.labels = [];

        if (!K3D.autoRendering || force) {
            if (renderingPromise === null) {
                renderingPromise = render().then(() => {
                    renderingPromise = null;
                });

                return renderingPromise;
            }
            if (force) {
                renderingPromise = renderingPromise.then(render).then(() => {
                    renderingPromise = null;
                });
            }
        }

        return renderingPromise;
    };

    this.renderOffScreen = function (width, height) {
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

        const rt = new THREE.WebGLRenderTarget(width, Math.ceil(height / chunkCount), {
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

        return getSSAAChunkedRender(self.renderer, self.axesHelper.scene, self.axesHelper.camera,
            rtAxesHelper, rtAxesHelper.width, rtAxesHelper.height, [[0, rtAxesHelper.height]],
            aaLevel, directRender).then((result) => {
            const axesHelper = new Uint8ClampedArray(width * height * 4);

            for (let y = 0; y < rtAxesHelper.height; y++) {
                // fast row-copy
                axesHelper.set(
                    result.slice(y * rtAxesHelper.width * 4, (y + 1) * rtAxesHelper.width * 4),
                    (y * width + width - rtAxesHelper.width) * 4,
                );
            }

            const rtGrid = chunkCount > 1
                ? new THREE.WebGLRenderTarget(width, height, { type: THREE.FloatType })
                : rt;

            return getSSAAChunkedRender(self.renderer, self.gridScene, self.camera,
                rtGrid, width, height, [[0, height]], aaLevel, directRender).then((grid) => {
                if (rtGrid !== rt) {
                    rtGrid.dispose();
                }

                K3D.parameters.clippingPlanes.forEach((plane) => {
                    self.renderer.clippingPlanes.push(new THREE.Plane(new THREE.Vector3().fromArray(plane), plane[3]));
                });

                computeAO(width, height);

                return getSSAAChunkedRender(self.renderer, self.scene, self.camera,
                    rt, width, height, chunkHeights,
                    aaLevel, currentRenderMethod).then((scene) => {
                    rt.dispose();
                    rtAxesHelper.dispose();
                    return [grid, scene, axesHelper];
                });
            });
        });
    };
};
