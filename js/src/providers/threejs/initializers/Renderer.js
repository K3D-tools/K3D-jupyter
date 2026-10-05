const THREE = require('three');
const cameraModes = require('../../../core/lib/cameraMode').cameraModes;
const error = require('../../../core/lib/Error').error;
const getSSAAChunkedRender = require('../helpers/SSAAChunkedRender');
const createCinematicPresenter = require('./cinematic/present');
const createDepthPeel = require('./passes/depthPeel');
const { depthOnBeforeCompile, colorOnBeforeCompile } = require('./passes/depthPeel');
const createAO = require('./passes/ao');
const { createAODepthMaterial } = require('./passes/ao');

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
    const planeGeometry = new THREE.PlaneGeometry(2, 2, 1, 1);
    const toneMappingMode = { value: 0 };
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
    const cameras = [];
    const fsCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    // the advanced renderer's ambient occlusion
    const ao = createAO(K3D, self, {
        globalPeelUniforms,
        depthMaterial,
        fsCamera,
        planeGeometry,
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

    const toneBlitScene = new THREE.Scene();

    {
        const plane = new THREE.Mesh(planeGeometry, toneBlitMaterial);

        plane.frustumCulled = false;
        toneBlitScene.add(plane);
    }

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

    const depthPeel = createDepthPeel(K3D, self, {
        gl,
        globalPeelUniforms,
        toneMappingMode,
        planeGeometry,
        depthMaterial,
        peelDummyNear,
        peelDummyFar,
        aoState: ao.state,
    });
    const depthPeelRender = depthPeel.render;

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
            ao.applyOverlay(camera, rt);

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
            ao.applyOverlay(camera, rt);
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

            ao.compute(size.x, size.y);

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

                ao.compute(width, height);

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
