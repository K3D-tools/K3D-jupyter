const THREE = require('three');
const { GTAOShader, generateMagicSquareNoise } = require('three/examples/jsm/shaders/GTAOShader.js');
const {
    PoissonDenoiseShader, generatePdSamplePointInitializer,
} = require('three/examples/jsm/shaders/PoissonDenoiseShader.js');
const cameraModes = require('../../../../core/lib/cameraMode').cameraModes;
const { depthOnBeforeCompile } = require('./depthPeel');

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
 * The advanced renderer's ambient occlusion: GTAO over a depth prepass, Poisson-denoised, then
 * multiplied onto the frame (applyOverlay) or onto each peel layer (state, read by depthPeel.js).
 * @param {Object} K3D current K3D instance
 * @param {Object} self the world the Renderer initializer runs on
 * @param {Object} shared resources of the raster pipeline
 */
module.exports = function createAO(K3D, self, shared) {
    const {
        globalPeelUniforms,
        depthMaterial,
        fsCamera,
        planeGeometry,
    } = shared;
    // AO prepass variant for wireframes: the wires occlude, the gaps between them do not
    const depthMaterialWireframe = new THREE.MeshDepthMaterial();

    // --- GTAO (advanced renderer only) ---
    // Full-frame AO computed once per frame/screenshot from a depth prepass (normals are
    // reconstructed from depth - no G-buffer), denoised spatially, then multiplied onto
    // every render of the main scene. Background depth == 1 is discarded by the shaders,
    // so the grid and the backdrop stay untouched.
    const aoTargets = { depth: null, raw: null, denoised: null };
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
        vertexShader: require('../shaders/composite.vertex.glsl'),
        fragmentShader: require('../shaders/aoOverlay.fragment.glsl'),
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

    const gtaoScene = new THREE.Scene();
    const pdScene = new THREE.Scene();
    const aoOverlayScene = new THREE.Scene();

    [[gtaoScene, gtaoMaterial], [pdScene, pdMaterial], [aoOverlayScene, aoOverlayMaterial]]
        .forEach(([scene, material]) => {
            const plane = new THREE.Mesh(planeGeometry, material);

            plane.frustumCulled = false;
            scene.add(plane);
        });

    depthMaterialWireframe.side = THREE.DoubleSide;
    depthMaterialWireframe.depthPacking = THREE.RGBADepthPacking;
    depthMaterialWireframe.wireframe = true;
    depthMaterialWireframe.onBeforeCompile = depthOnBeforeCompile.bind(null, globalPeelUniforms);
    depthMaterialWireframe.needsUpdate = true;

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

    // the AO buffers the peel composition multiplies each layer by, or null without AO
    function aoState() {
        return aoTexture === null ? null : {
            texture: aoTexture,
            volTexture: aoVolTexture,
            depthTexture: aoTargets.depth.texture,
        };
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

    return {
        compute: computeAO,
        applyOverlay: applyAOOverlay,
        state: aoState,
    };
};

module.exports.createAODepthMaterial = createAODepthMaterial;
