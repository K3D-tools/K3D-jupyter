const THREE = require('three');
const error = require('../../../../core/lib/Error').error;
const getSSAAChunkedRender = require('../../helpers/SSAAChunkedRender');
const { unpremultiply } = require('../../helpers/SSAAChunkedRender');
const alignAxesCamera = require('../../helpers/alignAxesCamera');
const cinematic = require('./index');
const createOIDN = require('./oidn');

/**
 * The cinematic renderer as Renderer.js sees it: what the canvas shows, the raster volume layer,
 * Open Image Denoise, and screenshots. The tracer and its loop live in ./index.js.
 * @param {Object} K3D current K3D instance
 * @param {Object} self the world the Renderer initializer runs on
 * @param {Object} shared resources of the raster pipeline this composes through
 */
module.exports = function createCinematicPresenter(K3D, self, shared) {
    const {
        fsCamera,
        planeGeometry,
        depthMaterial,
        globalPeelUniforms,
        peelDummyNear,
        toneBlitMaterial,
        toneBlitScene,
        directRender,
    } = shared;

    const presentClearColor = new THREE.Color();

    // lazy: building the path tracer and its BVH is expensive
    let cinematicMode = null;
    // Where the lens is focused, shown by cutting the scene open there rather than by standing a
    // plane in it. A plane cannot work: the raster volume draws with depthTest and depthWrite off
    // (objects/Volume.js:195), so it neither occludes nor is occluded, and a plane in front of it
    // reads as a flat backdrop at every distance. Clipping does work, because the volume shader
    // implements it inside its own ray march - so the cut face IS the focus plane, and dragging
    // the distance sweeps a cross-section through the data. Costs nothing in the traced image:
    // this is applied around the raster preview only.
    self.focusClipPlanes = function () {
        if (self.focusHelper !== true || K3D.parameters.renderer !== 'cinematic') {
            return null;
        }

        const distance = cinematic.resolveFocusDistance(K3D);
        const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(self.camera.quaternion);

        // keep what is at or beyond the focus distance: dot(forward, p) - (dot(forward, eye) + d) >= 0
        return [new THREE.Plane(
            forward,
            -(forward.dot(self.camera.position) + distance),
        )];
    };

    function presentCinematic(texture) {
        const buffer = new THREE.Vector2();
        const size = new THREE.Vector2();

        // drawing-buffer size, not CSS: gl_FragCoord in the blit spans device pixels
        self.renderer.getDrawingBufferSize(buffer);
        self.renderer.getSize(size);

        const clearAlpha = self.renderer.getClearAlpha();

        self.renderer.getClearColor(presentClearColor);
        // autoClear is off for the whole renderer and the blit composites premultiplied
        // over what the canvas already holds: presenting onto the previous frame walks a
        // semi-transparent object up to opaque, one sample at a time. Clearing costs
        // the axes, which are drawn again after the accumulation.
        self.renderer.setRenderTarget(null);
        self.renderer.setViewport(0, 0, size.x, size.y);
        // background_color is a CSS background on the target node, so the canvas
        // clears to nothing and lets it through
        self.renderer.setClearColor(0, 0);
        self.renderer.clear();

        composeCinematic(texture, null, buffer.x, buffer.y);

        self.renderer.setViewport(
            size.x - self.axesHelper.width,
            0,
            self.axesHelper.width,
            self.axesHelper.height,
        );
        self.renderer.render(self.axesHelper.scene, self.axesHelper.camera);
        self.renderer.setViewport(0, 0, size.x, size.y);
        self.renderer.setClearColor(presentClearColor, clearAlpha);
    }

    function getCinematic() {
        if (cinematicMode === null) {
            cinematicMode = cinematic(K3D, self.renderer, {
                // once per accumulation: the volumes march cut at the proxy depth
                prepareOverlay(proxyScene, width, height) {
                    lastProxyScene = proxyScene;
                    cinematicVolume.active = renderCinematicVolumeLayer(proxyScene, width, height);
                },

                // the accumulation reaches the canvas only via the shared compose/tone blit - one tone curve per mode
                presentFrame: presentCinematic,

                onConverged() {
                    denoiseConverged();
                },

                onError(e) {
                    error('Cinematic Error', `The cinematic renderer failed: ${e.message}.`, false);
                },

                // camera in motion: same scene, materials and environment, only the light transport is rasterised
                rasterizePreview() {
                    const size = new THREE.Vector2();

                    self.renderer.getSize(size);
                    self.renderer.setRenderTarget(null);
                    self.renderer.setViewport(0, 0, size.x, size.y);
                    self.renderer.clear();

                    const focusClip = self.focusClipPlanes();

                    if (focusClip !== null) {
                        self.renderer.clippingPlanes = focusClip;
                    }

                    directRender(self.scene, self.camera);

                    if (focusClip !== null) {
                        self.renderer.clippingPlanes = [];
                    }
                    self.renderer.setViewport(
                        size.x - self.axesHelper.width,
                        0,
                        self.axesHelper.width,
                        self.axesHelper.height,
                    );
                    self.renderer.render(self.axesHelper.scene, self.axesHelper.camera);
                    self.renderer.setViewport(0, 0, size.x, size.y);
                },
            });
        }

        return cinematicMode;
    }

    // internal hook for determinism and benchmark probes
    K3D.__cinematicSpike = getCinematic;

    // cinematic_denoise changed: on a finished image only the composition changes, so present it
    // again - denoising it first if that has not happened - instead of tracing one more sample,
    // which would make the denoised image stale and run the network again. False: render.
    self.recomposeCinematic = function () {
        if (cinematicMode === null || K3D.parameters.renderer !== 'cinematic') {
            return false;
        }

        const { key, target, converged } = cinematicMode.accumulation();

        if (!converged) {
            return false;
        }

        if ((K3D.parameters.cinematicDenoise || 0.0) > 0.0
            && (denoised === null || denoised.key !== key)) {
            denoiseConverged();

            return true;
        }

        presentCinematic(target.texture);

        return true;
    };

    // a concrete reason, or null; never a silent fallback to another renderer
    self.cinematicUnsupportedReason = function () {
        try {
            return getCinematic().unsupportedReason();
        } catch (e) {
            return e.message || 'cinematic initialization failed';
        }
    };

    // --- cinematic volume hybrid ---
    // MIPs (and volumes the tracer did not take over) stay out of the path-traced BVH: they
    // march from the camera to the first path-traced hit (proxy-scene depth, through the peel
    // segment uniforms) and composite premultiplied over the accumulation, before the tone curve.
    const cinematicVolume = { depth: null, layer: null, active: false };
    let composeTarget = null;
    // Open Image Denoise, once per converged accumulation (cinematic/oidn.js). Below 1 the
    // strength mixes the traced and the denoised image; 0 is off and leaves the trace untouched.
    const denoiseMixMaterial = new THREE.ShaderMaterial({
        uniforms: {
            tRaw: { value: null },
            tDenoised: { value: null },
            uMix: { value: 1.0 },
        },
        vertexShader: require('../shaders/composite.vertex.glsl'),
        fragmentShader: require('../shaders/denoiseMix.fragment.glsl'),
        depthTest: false,
        depthWrite: false,
        blending: THREE.NoBlending,
    });
    const denoiseMixScene = new THREE.Scene();
    let denoiseMixTarget = null;
    let oidn = null;
    // the last denoised accumulation, valid while the tracer still holds that accumulation
    let denoised = null;
    // the accumulation key of the pass in flight
    let denoising = null;
    // the scene the tracer was handed: the auxiliary buffers have to see the same geometry
    let lastProxyScene = null;
    const oidnWarnings = {};

    {
        const plane = new THREE.Mesh(planeGeometry, denoiseMixMaterial);

        plane.frustumCulled = false;
        denoiseMixScene.add(plane);
    }

    const rawBlitMaterial = new THREE.ShaderMaterial({
        uniforms: {
            tDiffuse: { value: null },
            uSize: { value: new THREE.Vector2(1, 1) },
            uPremultiply: { value: 0 },
        },
        vertexShader: require('../shaders/composite.vertex.glsl'),
        fragmentShader: require('../shaders/rawBlit.fragment.glsl'),
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: THREE.CustomBlending,
        blendEquation: THREE.AddEquation,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.OneMinusSrcAlphaFactor,
    });
    const rawBlitScene = new THREE.Scene();

    {
        const plane = new THREE.Mesh(planeGeometry, rawBlitMaterial);

        plane.frustumCulled = false;
        rawBlitScene.add(plane);
    }

    function ensureCinematicVolumeTargets(width, height) {
        if (cinematicVolume.depth !== null
            && cinematicVolume.depth.width === width
            && cinematicVolume.depth.height === height) {
            return;
        }

        if (cinematicVolume.depth !== null) {
            cinematicVolume.depth.dispose();
            cinematicVolume.layer.dispose();
        }

        // .r carries raw gl_FragCoord.z - the convention peelT reconstruction expects
        cinematicVolume.depth = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            format: THREE.RedFormat,
            type: THREE.FloatType,
        });
        cinematicVolume.layer = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            type: THREE.HalfFloatType,
            depthBuffer: false,
        });
    }

    // marches volumes/MIPs into the layer target, cut at proxy-scene depth; once per accumulation, not per sample
    function renderCinematicVolumeLayer(proxyScene, width, height) {
        const world = K3D.getWorld();
        const volumeObjects = [];
        // a volume the tracer tracks through is neither marched again nor a depth cut for the rest
        const traced = new Set();
        const tracedProxies = [];

        proxyScene.traverse((node) => {
            if (node.userData.k3dVolumeSource) {
                // a medium the tracer refused is a passthrough box: it must not cut the march,
                // and its source goes back to the raster layer, which is what the warning says
                if (node.userData.k3dVolumeRejected !== true) {
                    traced.add(node.userData.k3dVolumeSource);
                }

                tracedProxies.push(node);
            }
        });

        world.K3DObjects.traverse((obj) => {
            if (obj.visible && obj.userData.k3dVolumeShell && !traced.has(obj)) {
                volumeObjects.push(obj);
            }
        });

        if (volumeObjects.length === 0) {
            return false;
        }

        ensureCinematicVolumeTargets(width, height);

        // uLayer == 0 makes the peel tail a no-op, so leave uScreenSize alone:
        // ensureTargets skips rewriting it once the targets match
        globalPeelUniforms.uLayer.value = 0;
        self.camera.updateMatrixWorld();

        // overrideMaterial does not cover scene.background: env colours in the depth channel read as surfaces
        const savedBackground = proxyScene.background;

        const u = self.k3dVolumePeel;
        const shown = [];

        // the mutations below are global state the other renderers read - restore on throw
        try {
            proxyScene.background = null;
            tracedProxies.forEach((node) => {
                node.visible = false;
            });

            self.renderer.setRenderTarget(cinematicVolume.depth);
            self.renderer.setViewport(0, 0, width, height);
            self.renderer.setClearColor(0xffffff, 1);
            self.renderer.clear(true, true, false);
            proxyScene.overrideMaterial = depthMaterial;
            self.renderer.render(proxyScene, self.camera);

            u.uPeelSegment.value = 1;
            u.uPeelNearTexture.value = peelDummyNear;
            u.uPeelFarTexture.value = cinematicVolume.depth.texture;
            u.uPeelSize.value.set(1.0 / width, 1.0 / height);
            u.uPeelInvProjection.value.copy(self.camera.projectionMatrixInverse);
            u.uPeelInvView.value.copy(self.camera.matrixWorld);

            // leaves only - hiding the K3DObjects group would hide the volumes too
            world.K3DObjects.traverse((obj) => {
                if (obj.visible && obj.material && (!obj.userData.k3dVolumeShell || traced.has(obj))) {
                    obj.visible = false;
                    shown.push(obj);
                }
            });

            self.renderer.setRenderTarget(cinematicVolume.layer);
            self.renderer.setViewport(0, 0, width, height);
            self.renderer.setClearColor(0, 0);
            self.renderer.clear(true, false, false);
            self.renderer.render(self.scene, self.camera);
        } finally {
            proxyScene.overrideMaterial = null;
            proxyScene.background = savedBackground;
            tracedProxies.forEach((node) => {
                node.visible = true;
            });
            shown.forEach((obj) => {
                obj.visible = true;
            });
            u.uPeelSegment.value = 0;
            self.renderer.setRenderTarget(null);
        }

        return true;
    }

    function warnOIDN(reason) {
        if (!oidnWarnings[reason]) {
            oidnWarnings[reason] = true;

            console.warn(`K3D cinematic: the image is shown without denoising - ${reason}.`);
        }
    }

    function getOIDN() {
        if (oidn === null) {
            oidn = createOIDN();
        }

        return oidn;
    }

    // Rasterised views averaged over the lens, so that the guides blur where the trace blurs:
    // sharp albedo under a defocused background makes OIDN draw edges the image does not have.
    const AUX_LENS_SAMPLES = 128;

    // The aperture points of upstream's camera ray (sampleCircle / sampleRegularPolygon), on an
    // R2 sequence instead of random draws, in scene units: the radius is half of the bokeh size.
    function lensSamples(radius, blades, count) {
        const samples = [];
        const g = 1.2207440846057596;
        const a1 = 1 / g;
        const a2 = 1 / (g * g);
        const a3 = 1 / (g * g * g);

        for (let i = 0; i < count; i++) {
            const u = (0.5 + a1 * (i + 1)) % 1;
            const v = (0.5 + a2 * (i + 1)) % 1;
            const w = (0.5 + a3 * (i + 1)) % 1;
            let x;
            let y;

            if (blades >= 3) {
                const step = (2 * Math.PI) / blades;
                const angle1 = step * Math.floor(blades * u);
                const angle2 = angle1 + step;
                let r1 = v;
                let r2 = w;

                if (r1 + r2 > 1) {
                    r1 = 1 - r1;
                    r2 = 1 - r2;
                }

                x = Math.sin(angle1) * r1 + Math.sin(angle2) * r2;
                y = Math.cos(angle1) * r1 + Math.cos(angle2) * r2;
            } else {
                x = Math.cos(2 * Math.PI * u) * Math.sqrt(v);
                y = Math.sin(2 * Math.PI * u) * Math.sqrt(v);
            }

            samples.push([x * radius, y * radius]);
        }

        return samples;
    }

    // The camera moved across the lens by (ox, oy), its frustum sheared so that the plane at the
    // focus distance stays where it was. The tracer focuses on a sphere around the eye rather than
    // a plane; the two agree on the axis and part a little towards the corners.
    function lensCamera(camera, ox, oy, focus) {
        const lens = camera.clone();
        const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
        const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);

        lens.position.addScaledVector(right, ox).addScaledVector(up, oy);
        lens.updateMatrixWorld(true);
        lens.projectionMatrix.copy(camera.projectionMatrix);

        const m = lens.projectionMatrix.elements;

        m[8] -= (m[0] * ox) / focus;
        m[9] -= (m[5] * oy) / focus;
        lens.projectionMatrixInverse.copy(lens.projectionMatrix).invert();

        return lens;
    }

    // inside a participating medium there is no first surface for albedo and normals to describe
    function hasMedium() {
        const objects = K3D.getWorld().ObjectsListJson;

        return Object.keys(objects).some((id) => {
            const json = objects[id];

            return (json.type === 'Volume' || json.type === 'MIP') && json.visible !== false;
        });
    }

    // Albedo and normals of the first surface, which OIDN uses to keep edges and textures:
    // the scene the tracer was given, rasterised twice with its materials swapped.
    function renderAuxBuffers(width, height) {
        if (lastProxyScene === null || hasMedium()) {
            return null;
        }

        const meshes = [];

        lastProxyScene.traverse((object) => {
            if (object.isMesh && object.material && !Array.isArray(object.material)) {
                meshes.push([object, object.material]);
            }
        });

        const passes = [
            {
                // linear, as the target is not sRGB: OIDN wants albedo in linear [0, 1]
                clear: new THREE.Color(0, 0, 0),
                make: (m) => new THREE.MeshBasicMaterial({
                    color: m.color ? m.color.clone() : new THREE.Color(1, 1, 1),
                    map: m.map || null,
                    vertexColors: Boolean(m.vertexColors),
                    side: m.side,
                    alphaTest: m.alphaTest || 0,
                    toneMapped: false,
                }),
            },
            {
                // n * 0.5 + 0.5 in view space; a pixel with no surface has a zero normal
                clear: new THREE.Color().setRGB(0.5, 0.5, 0.5, THREE.LinearSRGBColorSpace),
                make: (m) => new THREE.MeshNormalMaterial({
                    side: m.side,
                    normalMap: m.normalMap || null,
                    normalScale: m.normalScale ? m.normalScale.clone() : new THREE.Vector2(1, 1),
                    flatShading: Boolean(m.flatShading),
                }),
            },
        ];
        // the lens as the tracer reads it off the camera, millimetres included - not the plot
        // parameter, which reaches the camera through an fStop the focal length then rescales
        const lens = self.camera;
        const views = (K3D.parameters.cinematicBokehSize || 0.0) > 0.0 && lens.bokehSize > 0.0
            ? lensSamples(lens.bokehSize * 0.5e-3, lens.apertureBlades || 0, AUX_LENS_SAMPLES)
                .map(([ox, oy]) => lensCamera(lens, ox, oy, lens.focusDistance))
            : [self.camera];
        const previousTarget = self.renderer.getRenderTarget();
        const previousClear = new THREE.Color();
        const previousAlpha = self.renderer.getClearAlpha();
        const background = lastProxyScene.background;
        const target = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
        });
        const out = [];

        self.renderer.getClearColor(previousClear);
        lastProxyScene.background = null;

        try {
            passes.forEach((pass) => {
                const materials = meshes.map(([object, material]) => {
                    const swapped = pass.make(material);

                    object.material = swapped;

                    return swapped;
                });

                const pixels = new Uint8Array(width * height * 4);
                const sum = views.length > 1 ? new Float32Array(width * height * 4) : null;

                views.forEach((view) => {
                    self.renderer.setRenderTarget(target);
                    self.renderer.setViewport(0, 0, width, height);
                    self.renderer.setClearColor(pass.clear, 1);
                    self.renderer.clear();
                    self.renderer.render(lastProxyScene, view);
                    self.renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);

                    if (sum !== null) {
                        for (let i = 0; i < pixels.length; i++) {
                            sum[i] += pixels[i];
                        }
                    }
                });

                if (sum === null) {
                    out.push(new Uint8ClampedArray(pixels.buffer));
                } else {
                    // Uint8ClampedArray rounds on assignment
                    out.push(Uint8ClampedArray.from(sum, (value) => value / views.length));
                }

                materials.forEach((material) => material.dispose());
            });
        } finally {
            meshes.forEach(([object, material]) => {
                object.material = material;
            });
            lastProxyScene.background = background;
            self.renderer.setClearColor(previousClear, previousAlpha);
            self.renderer.setRenderTarget(previousTarget);
            target.dispose();
        }

        return { albedo: out[0], normal: out[1] };
    }

    // resolves with the denoised accumulation as a float texture, or null
    function runOIDN(target, width, height, onProgress) {
        const engine = getOIDN();
        const reason = engine.unavailableReason();

        if (reason !== null) {
            warnOIDN(reason);

            return Promise.resolve(null);
        }

        // read now: the next sample overwrites the target
        const color = new Float32Array(width * height * 4);

        self.renderer.readRenderTargetPixels(target, 0, 0, width, height, color);

        const aux = renderAuxBuffers(width, height);

        return engine.denoise(color, aux && aux.albedo, aux && aux.normal, width, height, onProgress)
            .then((pixels) => {
                if (pixels === null) {
                    return null;
                }

                const texture = new THREE.DataTexture(
                    pixels,
                    width,
                    height,
                    THREE.RGBAFormat,
                    THREE.FloatType,
                );

                texture.minFilter = THREE.NearestFilter;
                texture.magFilter = THREE.NearestFilter;
                texture.needsUpdate = true;

                return texture;
            }, (e) => {
                warnOIDN(e.message || String(e));

                return null;
            });
    }

    function replaceDenoised(next) {
        if (denoised !== null) {
            denoised.texture.dispose();
        }

        denoised = next;
    }

    // the interactive loop reached its budget: denoise what it shows, then show that instead
    function denoiseConverged() {
        if (!((K3D.parameters.cinematicDenoise || 0.0) > 0.0)) {
            return;
        }

        const mode = getCinematic();
        const { key, target } = mode.accumulation();

        if ((denoised !== null && denoised.key === key && !denoised.partial) || denoising === key) {
            return;
        }

        if (denoising !== null) {
            getOIDN().cancel();
        }

        denoising = key;
        mode.setHud('cinematic: denoising…');

        const { width, height } = target;
        let preview = null;

        // every finished tile is shown as it lands, over the traced image
        function onProgress(pixels, done, total) {
            if (mode.accumulation().key !== key) {
                return;
            }

            if (preview === null) {
                preview = new THREE.DataTexture(
                    pixels,
                    width,
                    height,
                    THREE.RGBAFormat,
                    THREE.FloatType,
                );
                preview.minFilter = THREE.NearestFilter;
                preview.magFilter = THREE.NearestFilter;
                replaceDenoised({
                    key, texture: preview, width, height, partial: true,
                });
            }

            preview.needsUpdate = true;
            mode.setHud(`cinematic: denoising ${done} / ${total}`);
            presentCinematic(target.texture);
        }

        runOIDN(target, width, height, onProgress).then((texture) => {
            if (denoising === key) {
                denoising = null;
            }

            if (texture === null || mode.accumulation().key !== key) {
                if (texture !== null) {
                    texture.dispose();
                }

                return;
            }

            replaceDenoised({
                key, texture, width, height,
            });
            mode.setHud(null);
            presentCinematic(target.texture);
        });
    }

    // compose accumulation and volume layer in linear space, then exactly one tone curve
    // The one place the traced image can be filtered: after the accumulation, before tone
    // mapping, and on the path both the canvas and the screenshot go through.
    function denoiseCinematic(ptTexture, width, height, override) {
        const strength = K3D.parameters.cinematicDenoise || 0.0;

        // zero is off, and the only value that leaves the traced image exactly as it was
        if (!(strength > 0.0)) {
            return ptTexture;
        }

        let texture = override || null;

        if (texture === null && denoised !== null && denoised.width === width
            && denoised.height === height && getCinematic().accumulation().key === denoised.key) {
            texture = denoised.texture;
        }

        // still accumulating, or nothing to denoise with: the trace as it is
        if (texture === null) {
            return ptTexture;
        }

        if (strength >= 1.0) {
            return texture;
        }

        if (denoiseMixTarget === null || denoiseMixTarget.width !== width
            || denoiseMixTarget.height !== height) {
            if (denoiseMixTarget !== null) {
                denoiseMixTarget.dispose();
            }

            denoiseMixTarget = new THREE.WebGLRenderTarget(width, height, {
                minFilter: THREE.NearestFilter,
                magFilter: THREE.NearestFilter,
                type: THREE.FloatType,
                depthBuffer: false,
            });
        }

        const u = denoiseMixMaterial.uniforms;

        u.tRaw.value = ptTexture;
        u.tDenoised.value = texture;
        u.uMix.value = strength;

        self.renderer.setRenderTarget(denoiseMixTarget);
        self.renderer.setViewport(0, 0, width, height);
        self.renderer.render(denoiseMixScene, fsCamera);

        u.tRaw.value = null;
        u.tDenoised.value = null;

        return denoiseMixTarget.texture;
    }

    function composeCinematic(rawTexture, rt, width, height, denoisedTexture) {
        const ptTexture = denoiseCinematic(rawTexture, width, height, denoisedTexture);

        if (!cinematicVolume.active) {
            toneBlitMaterial.uniforms.tDiffuse.value = ptTexture;
            toneBlitMaterial.uniforms.uSize.value.set(width, height);
            toneBlitMaterial.uniforms.uPremultiplied.value = 0;

            self.renderer.setRenderTarget(rt);
            self.renderer.setViewport(0, 0, width, height);
            self.renderer.render(toneBlitScene, fsCamera);
            return;
        }

        if (composeTarget === null || composeTarget.width !== width || composeTarget.height !== height) {
            if (composeTarget !== null) {
                composeTarget.dispose();
            }

            composeTarget = new THREE.WebGLRenderTarget(width, height, {
                minFilter: THREE.NearestFilter,
                magFilter: THREE.NearestFilter,
                type: THREE.HalfFloatType,
                depthBuffer: false,
            });
        }

        self.renderer.setRenderTarget(composeTarget);
        self.renderer.setViewport(0, 0, width, height);
        self.renderer.setClearColor(0, 0);
        self.renderer.clear(true, false, false);

        rawBlitMaterial.uniforms.uSize.value.set(width, height);
        // upstream's BlendMaterial writes straight colour with coverage in alpha, the layer comes
        // out of ordinary blending premultiplied, and the blend below composites premultiplied
        rawBlitMaterial.uniforms.uPremultiply.value = 1;
        rawBlitMaterial.uniforms.tDiffuse.value = ptTexture;
        self.renderer.render(rawBlitScene, fsCamera);

        rawBlitMaterial.uniforms.uPremultiply.value = 0;
        rawBlitMaterial.uniforms.tDiffuse.value = cinematicVolume.layer.texture;
        self.renderer.render(rawBlitScene, fsCamera);

        toneBlitMaterial.uniforms.tDiffuse.value = composeTarget.texture;
        toneBlitMaterial.uniforms.uSize.value.set(width, height);
        toneBlitMaterial.uniforms.uPremultiplied.value = 1;

        self.renderer.setRenderTarget(rt);
        self.renderer.setViewport(0, 0, width, height);
        self.renderer.render(toneBlitScene, fsCamera);
    }

    // accumulates the sample budget at the target resolution, tone-maps through the
    // shared blit and reads back. No grid layer - the environment dome is the background.
    function cinematicOffScreen(width, height) {
        const reason = self.cinematicUnsupportedReason();

        if (reason !== null) {
            error('Cinematic Error', `The cinematic renderer cannot start: ${reason}.`, false);

            return Promise.resolve([]);
        }

        const mode = getCinematic();
        // the interactive loop shares this tracer: left running, its next frame resizes the tiles
        // and turns per-tile blending back on in the middle of this accumulation
        const resume = mode.isRunning();

        mode.abort();

        const size = new THREE.Vector2();

        // per-frame object work (volume light maps included) hangs off BEFORE_RENDER
        K3D.refreshGrid();
        self.camera.updateMatrixWorld();
        alignAxesCamera(self, K3D);
        K3D.dispatch(K3D.events.BEFORE_RENDER);
        self.renderer.getSize(size);

        const scale = Math.max(width / size.x, height / size.y);
        const aaLevel = Math.max(Math.min(5, K3D.parameters.antialias), 0);
        const rtAxesHelper = new THREE.WebGLRenderTarget(
            self.axesHelper.width * scale,
            self.axesHelper.height * scale,
            { type: THREE.FloatType },
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
                axesHelper.set(
                    result.slice(y * rtAxesHelper.width * 4, (y + 1) * rtAxesHelper.width * 4),
                    (y * width + width - rtAxesHelper.width) * 4,
                );
            }
            rtAxesHelper.dispose();

            // the tracer stays pinned to this resolution until released - release on any outcome
            return mode.renderBudget(width, height).then((frame) => {
                if (!((K3D.parameters.cinematicDenoise || 0.0) > 0.0)) {
                    return { frame, denoisedTexture: null };
                }

                return runOIDN(mode.accumulation().target, width, height)
                    .then((denoisedTexture) => ({ frame, denoisedTexture }));
            }).then(({ frame, denoisedTexture }) => {
                const rt = new THREE.WebGLRenderTarget(width, height, {
                    minFilter: THREE.NearestFilter,
                    magFilter: THREE.NearestFilter,
                });

                try {
                    self.renderer.setRenderTarget(rt);
                    self.renderer.setViewport(0, 0, width, height);
                    self.renderer.setClearColor(0, 0);
                    self.renderer.clear();
                    composeCinematic(frame.texture, rt, width, height, denoisedTexture);

                    const pixels = new Uint8Array(width * height * 4);

                    self.renderer.readRenderTargetPixels(rt, 0, 0, width, height, pixels);

                    return [unpremultiply(new Uint8ClampedArray(pixels.buffer)), axesHelper];
                } finally {
                    self.renderer.setRenderTarget(null);
                    rt.dispose();

                    if (denoisedTexture !== null) {
                        denoisedTexture.dispose();
                    }
                }
            }).finally(() => {
                mode.releaseFixedSize();

                // releaseFixedSize resets the tracer, so there is nothing left to keep: the
                // viewport starts a fresh accumulation instead of freezing on the last frame
                if (resume) {
                    mode.wake();
                }
            });
        });
    }

    // the cinematic branch of Renderer.render()
    function render() {
        const reason = self.cinematicUnsupportedReason();

        if (reason !== null) {
            error('Cinematic Error', `The cinematic renderer cannot start: ${reason}.`, false);

            return Promise.resolve(null);
        }

        const mode = getCinematic();

        self.renderer.clippingPlanes = [];
        // DOM overlays reproject here; required before the first frame or they stay stale
        K3D.refreshGrid();
        self.camera.updateMatrixWorld();
        alignAxesCamera(self, K3D);
        K3D.dispatch(K3D.events.BEFORE_RENDER);

        // interactive accumulation runs in its own animation-frame loop: it keeps
        // refining after this returns and dispatches RENDERED itself
        if (typeof window !== 'undefined' && typeof window.headlessK3D === 'undefined') {
            mode.wake();

            return Promise.resolve(null);
        }

        return mode.renderFrame().then((result) => {
            if (!result.stale) {
                K3D.dispatch(K3D.events.RENDERED);
            }

            return result;
        }).catch((e) => {
            error('Cinematic Error', `The cinematic renderer failed: ${e.message}.`, false);

            return null;
        });
    }

    return {
        render,
        offScreen: cinematicOffScreen,

        // leaving cinematic: an in-flight accumulation would paint over the raster frames
        leave() {
            if (cinematicMode !== null) {
                cinematicMode.abort();
                cinematicMode.hideHud();
            }
        },

        abort() {
            if (cinematicMode !== null) {
                cinematicMode.abort();
            }
        },
    };
};
