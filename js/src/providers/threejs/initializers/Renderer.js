const THREE = require('three');
const cameraModes = require('../../../core/lib/cameraMode').cameraModes;
const error = require('../../../core/lib/Error').error;
const getSSAAChunkedRender = require('../helpers/SSAAChunkedRender');

function depthOnBeforeCompile(globalPeelUniforms, shader) {
    shader.uniforms.uScreenSize = globalPeelUniforms.uScreenSize;
    shader.uniforms.uPrevDepthTexture = globalPeelUniforms.uPrevDepthTexture;
    shader.uniforms.uLayer = globalPeelUniforms.uLayer;
    shader.uniforms.uDepthOffset = globalPeelUniforms.uDepthOffset;

    shader.fragmentShader = require('./shaders/depthShader.fragment.header.glsl') + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace(/}$/gm, require('./shaders/depthShader.fragment.tail.glsl'));
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
    let depthStencilBuffer;
    const compositeScene = new THREE.Scene();
    const planeGeometry = new THREE.PlaneBufferGeometry(2, 2, 1, 1);
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

    if (!context) {
        if (typeof WebGL2RenderingContext !== 'undefined') {
            error(
                'Your browser appears to support WebGL2 but it might '
                + 'be disabled. Try updating your OS and/or video card driver.',
                true,
            );
        } else {
            error(
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

    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    console.log('K3D: (UNMASKED_VENDOR_WEBGL)', gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL));
    console.log('K3D: (UNMASKED_RENDERER_WEBGL)', gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL));
    console.log('K3D: (depth bits)', gl.getParameter(gl.DEPTH_BITS));
    console.log('K3D: (stencil bits)', gl.getParameter(gl.STENCIL_BITS));

    function ensureTargets(depthBuffer, width, height) {
        if (targets.length > 0
            && targets[0].width === width
            && targets[0].height === height) {
            return;
        }

        globalPeelUniforms.uScreenSize.value.set(1 / width, 1 / height);

        if (targets.length) {
            for (let i = 0; i < 3; i++) {
                targets.pop().dispose();
            }
        }

        for (let i = 0; i < 3; i++) {
            targets.push(
                new THREE.WebGLRenderTarget(
                    width,
                    height,
                    {
                        minFilter: THREE.NearestFilter,
                        magFilter: THREE.NearestFilter,
                    },
                ),
            );

            targets[i].ownDepthBuffer = depthBuffer;
        }
    }

    function advancedRender(scene, camera, rt) {
        if (typeof (rt) === 'undefined') {
            rt = null;
            ensureTargets(depthStencilBuffer, K3D.getWorld().width, K3D.getWorld().height);
        } else {
            ensureTargets(depthStencilBuffer, rt.width, rt.height);
        }

        K3D.getWorld().K3DObjects.children.forEach((obj) => {
            if (obj.material && obj.material.opacity <= 0.0) {
                obj.visible = false;
            }
        });

        globalPeelUniforms.uLayer.value = 0;
        globalPeelUniforms.uPrevDepthTexture.value = null;
        compositeMaterial.uniforms.uBlit.value = 1;

        // STAGE I - clear
        gl.colorMask(true, true, true, true);
        gl.depthMask(true);

        // self.renderer.setRenderTarget(null);
        // self.renderer.setClearColor(0xffffff, 0);
        // self.renderer.clear();

        self.renderer.setRenderTarget(targets[0]);
        self.renderer.setClearColor(0, 0);
        self.renderer.clear();

        self.renderer.setRenderTarget(targets[1]);
        self.renderer.setClearColor(0xffffff, 1);
        self.renderer.clear();

        self.renderer.setRenderTarget(targets[2]);
        self.renderer.setClearColor(0, 0);
        self.renderer.clear();

        // STAGE II - enable stencil and color
        if (K3D.parameters.depthPeels < 8) {
            gl.enable(gl.STENCIL_TEST);
        }

        gl.colorMask(true, true, true, true);
        gl.stencilFunc(gl.ALWAYS, 1, 0xff);
        gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);

        scene.overrideMaterial = depthMaterial;

        self.renderer.setRenderTarget(targets[1]);
        self.renderer.render(scene, camera);

        // STAGE III
        scene.overrideMaterial = null;

        gl.stencilFunc(gl.EQUAL, 1, 0xff);
        gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);

        self.renderer.setRenderTarget(targets[0]);
        self.renderer.clear(false, true, false);
        self.renderer.render(scene, camera);

        // STAGE IV
        compositeMaterial.blendSrc = THREE.OneMinusDstAlphaFactor;
        compositeMaterial.blendDst = THREE.OneFactor;

        compositeMaterial.uniforms.uTextureA.value = targets[0].texture;
        compositeMaterial.uniforms.uBlit.value = 1;

        gl.stencilFunc(gl.EQUAL, 1, 0xff);
        gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);

        self.renderer.setRenderTarget(targets[2]);
        self.renderer.render(compositeScene, camera);

        // STAGE V
        let bit = 1;

        for (let i = 0; i < K3D.parameters.depthPeels; i++) {
            // continue
            const flip = i % 2;
            const flop = (i + 1) % 2;

            // next peel
            globalPeelUniforms.uPrevDepthTexture.value = targets[flop].texture;
            globalPeelUniforms.uLayer.value = i + 1;

            self.renderer.setRenderTarget(targets[flip]);
            self.renderer.setClearColor(0, 0);
            self.renderer.clear(true, true, false);

            bit |= 1 << (i + 1);

            gl.stencilFunc(gl.EQUAL, bit, 1 << i);
            gl.colorMask(true, true, true, true);

            // replace stencil with next level
            gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);

            // render color into target
            self.renderer.setRenderTarget(targets[flip]);
            self.renderer.render(scene, camera);

            // blit to 3rd buffer
            compositeMaterial.uniforms.uTextureA.value = targets[flip].texture;

            gl.stencilFunc(gl.EQUAL, bit, 1 << (i + 1));
            gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);

            self.renderer.setRenderTarget(targets[2]);
            self.renderer.render(compositeScene, camera);

            // clear depth target and render
            self.renderer.setRenderTarget(targets[flip]);
            self.renderer.setClearColor(0xffffff, 1);
            self.renderer.clear(true, true, false);

            scene.overrideMaterial = depthMaterial;

            gl.stencilFunc(gl.EQUAL, bit, 1 << (i + 1));
            gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);

            self.renderer.setRenderTarget(targets[flip]);
            self.renderer.render(scene, camera);

            scene.overrideMaterial = null;
        }

        // STAGE VI
        gl.disable(gl.STENCIL_TEST);

        globalPeelUniforms.uLayer.value = 0;
        gl.stencilFunc(gl.ALWAYS, 1, 0xff);
        gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);

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

    function standardRender(scene, camera, rt) {
        if (typeof (rt) === 'undefined') {
            rt = null;
        }

        self.renderer.setRenderTarget(rt);
        self.renderer.render(scene, camera);
    }

    function render() {
        const currentRenderMethod = K3D.parameters.depthPeels > 0 ? advancedRender : standardRender;

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

                if (K3D.autoRendering) {
                    requestAnimationFrame(render);
                } else {
                    resolve(true);
                }
            });

            resolve(null);
        });
    }

    compositePlane.frustumCulled = false;
    compositeScene.add(compositePlane);

    depthMaterial.side = THREE.DoubleSide;
    depthMaterial.depthPacking = THREE.RGBADepthPacking;
    depthMaterial.onBeforeCompile = depthOnBeforeCompile.bind(null, globalPeelUniforms);
    depthMaterial.needsUpdate = true;

    depthStencilBuffer = gl.createRenderbuffer();

    this.renderer.setClearColor(0, 0);
    this.renderer.autoClear = false;

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
        const currentRenderMethod = K3D.parameters.depthPeels > 0 ? advancedRender : standardRender;

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
            aaLevel, standardRender).then((result) => {
            const axesHelper = new Uint8ClampedArray(width * height * 4);

            for (let y = 0; y < rtAxesHelper.height; y++) {
                // fast row-copy
                axesHelper.set(
                    result.slice(y * rtAxesHelper.width * 4, (y + 1) * rtAxesHelper.width * 4),
                    (y * width + width - rtAxesHelper.width) * 4,
                );
            }

            return getSSAAChunkedRender(self.renderer, self.gridScene, self.camera,
                rt, width, height, [[0, height]], aaLevel, standardRender).then((grid) => {
                K3D.parameters.clippingPlanes.forEach((plane) => {
                    self.renderer.clippingPlanes.push(new THREE.Plane(new THREE.Vector3().fromArray(plane), plane[3]));
                });

                return getSSAAChunkedRender(self.renderer, self.scene, self.camera,
                    rt, width, height, chunkHeights,
                    aaLevel, currentRenderMethod).then((scene) => {
                    rt.dispose();
                    return [grid, scene, axesHelper];
                });
            });
        });
    };
};
