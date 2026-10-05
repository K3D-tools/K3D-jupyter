const THREE = require('three');

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
 * Depth peeling: the scene in layers front to back, composited into one accumulator, with volumes
 * marched in the segments between the layers (#277). Orthogonal to the renderer mode.
 * @param {Object} K3D current K3D instance
 * @param {Object} self the world the Renderer initializer runs on
 * @param {Object} shared resources of the raster pipeline; aoState() is null without AO
 */
module.exports = function createDepthPeel(K3D, self, shared) {
    const {
        gl,
        globalPeelUniforms,
        toneMappingMode,
        planeGeometry,
        depthMaterial,
        peelDummyNear,
        peelDummyFar,
        aoState,
    } = shared;
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
        vertexShader: require('../shaders/composite.vertex.glsl'),
        fragmentShader: require('../shaders/composite.fragment.glsl'),
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
    const compositePlane = new THREE.Mesh(planeGeometry, compositeMaterial);

    compositePlane.frustumCulled = false;
    compositeScene.add(compositePlane);

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
        const ao = aoState();

        compositeMaterial.uniforms.uAoEnabled.value = ao !== null ? 1 : 0;

        if (ao !== null) {
            compositeMaterial.uniforms.tAO.value = ao.texture;
            compositeMaterial.uniforms.tAOVol.value = ao.volTexture;
            compositeMaterial.uniforms.tAODepth.value = ao.depthTexture;

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

    return {
        render: depthPeelRender,
    };
};

module.exports.depthOnBeforeCompile = depthOnBeforeCompile;
module.exports.colorOnBeforeCompile = colorOnBeforeCompile;
