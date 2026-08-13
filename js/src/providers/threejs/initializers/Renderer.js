const THREE = require('three');
const cameraModes = require('../../../core/lib/cameraMode').cameraModes;
const error = require('../../../core/lib/Error').error;
const getSSAAChunkedRender = require('../helpers/SSAAChunkedRender');

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
    const compositeMaterial = new THREE.ShaderMaterial({
        uniforms: {
            uTextureA: { value: null },
            uTextureB: { value: null },
            uBlit: { value: 0 },
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

        self.renderer.setRenderTarget(rt);
        self.renderer.render(scene, camera);
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

    self.applyToneMapping = function (name) {
        const map = {
            none: THREE.NoToneMapping,
            agx: THREE.AgXToneMapping,
            aces: THREE.ACESFilmicToneMapping,
        };

        self.renderer.toneMapping = map[name] || THREE.NoToneMapping;

        // a toneMapping change recompiles nothing on its own
        self.K3DObjects.traverse((obj) => {
            if (obj.material) {
                obj.material.needsUpdate = true;
            }
        });
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
