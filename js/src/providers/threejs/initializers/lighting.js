const THREE = require('three');
const environmentHelper = require('../helpers/environment');

/**
 * Lighting of the three renderer modes: the simple renderer's light kit, and for advanced and
 * cinematic the environment map with its spherical-harmonics digest for the bespoke-light
 * shaders. Installs recalculateLights() and applyRendererMode() on the world.
 * @param {Object} K3D current K3D instance
 * @param {Object} self the world the Scene initializer runs on; its scene and camera exist
 */
module.exports = function installLighting(K3D, self) {
    const initialLightIntensity = {
        ambient: 0.2 * Math.PI,
        key: 0.4 * Math.PI,
        head: 0.15 * Math.PI,
        fill: 0.15 * Math.PI,
        back: 0.1 * Math.PI,
    };
    const unlitAmbient = Math.PI;
    const ambientLight = new THREE.AmbientLight(0xffffff);

    // shared by reference with the bespoke-light shaders (volume, mip, points 3d):
    // zero in simple, the environment's SH radiance in advanced. The L1 band is
    // carried separately as one directional light (dir + colour), and the rotation
    // maps world-space normals into env space, same convention as envMapRotation.
    self.k3dEnvSH = {
        value: Array.from({ length: 9 }, () => new THREE.Vector3(0, 0, 0)),
    };
    self.k3dEnvRotation = { value: new THREE.Matrix3() };
    self.k3dEnvLightDir = { value: new THREE.Vector3(0, 0, 1) };
    self.k3dEnvLightColor = { value: new THREE.Vector3(0, 0, 0) };
    // the same surface-delivery correction that scene.environmentIntensity carries
    // for PMREM materials; SH-lit SURFACES (points 3d impostor) apply it, volumes
    // (calibrated at parity without it) do not
    self.k3dEnvSurfaceBoost = { value: 1.2 };

    // https://www.vtk.org/doc/release/5.0/html/a01682.html
    // A LightKit consists of three lights, a key light, a fill light, and a headlight. The main light is the key
    // light. It is usually positioned so that it appears like an overhead light (like the sun, or a ceiling light).
    // It is generally positioned to shine down on the scene from about a 45 degree angle vertically and at least a
    // little offset side to side. The key light usually at least about twice as bright as the total of all other
    // lights in the scene to provide good modeling of object features.

    // The other lights in the kit (the fill light, headlight, and a pair of back lights) are weaker sources that
    // provide extra illumination to fill in the spots that the key light misses. The fill light is usually
    // positioned across from or opposite from the key light (though still on the same side of the object as the
    // camera) in order to simulate diffuse reflections from other objects in the scene. The headlight, always
    // located at the position of the camera, reduces the contrast between areas lit by the key and fill light.
    // The two back lights, one on the left of the object as seen from the observer and one on the right, fill on
    // the high-contrast areas behind the object. To enforce the relationship between the different lights, the
    // intensity of the fill, back and headlights are set as a ratio to the key light brightness. Thus, the
    // brightness of all the lights in the scene can be changed by changing the key light intensity.

    self.keyLight = new THREE.DirectionalLight(0xffffff); // key
    self.headLight = new THREE.DirectionalLight(0xffffff); // head
    self.fillLight = new THREE.DirectionalLight(0xffffff); // fill
    self.backLight = new THREE.DirectionalLight(0xffffff); // back

    self.keyLight.position.set(0.25, 1, 1.0);
    self.headLight.position.set(0, 0, 1);
    self.fillLight.position.set(-0.25, -1, 1.0);
    self.backLight.position.set(-2.5, 0.4, -1);

    [self.keyLight, self.headLight, self.fillLight, self.backLight].forEach((light) => {
        self.camera.add(light);
        self.camera.add(light.target);
        // self.scene.add(new THREE.DirectionalLightHelper(light, 1.0, 0xff0000));
    });

    self.scene.add(ambientLight);

    self.recalculateLights = function (value) {
        if (K3D.parameters.renderer === 'advanced' || K3D.parameters.renderer === 'cinematic') {
            // The environment is the only light: the maps are normalised to the mean
            // delivery of the whole simple rig, so switching modes changes the light
            // direction, not the exposure.
            ambientLight.intensity = value <= 1.0 ? unlitAmbient * (1.0 - value) : 0.0;
            // above 1 the simple rig stops scaling ambient, so the environment follows
            // the same knee to keep the mean delivery of both modes equal at every value
            const envIntensity = value <= 1.0 ? Math.max(value, 0.0) : (1.0 + value) / 2.0;

            // The old rig chased the camera, so visible surfaces got more than the
            // sphere mean the maps are normalised to. Measured on the reference base:
            // standard materials land at 0.87 of simple, the SH consumers (volume,
            // mip) at ~1.0 - so only the surface path gets the correction.
            self.scene.environmentIntensity = envIntensity * 1.2;

            const sh = (environmentEquirect && environmentEquirect.userData.k3dSH) || null;
            const rotation = new THREE.Matrix4().makeRotationFromEuler(environmentRotation(K3D));

            self.k3dEnvRotation.value.setFromMatrix4(rotation).transpose();

            // bands 1-3 (L1) are zeroed here and travel as the directional light below
            for (let b = 0; b < 9; b++) {
                if (sh && (b < 1 || b > 3)) {
                    self.k3dEnvSH.value[b].set(sh[b * 3], sh[b * 3 + 1], sh[b * 3 + 2])
                        .multiplyScalar(envIntensity);
                } else {
                    self.k3dEnvSH.value[b].set(0, 0, 0);
                }
            }

            // the dominant directional light: direction is the luminance-weighted L1
            // vector in env space (three's basis order: band 1 = y, 2 = z, 3 = x),
            // colour is the L1 irradiance evaluated at that direction
            self.k3dEnvLightColor.value.set(0, 0, 0);

            if (sh) {
                const dir = new THREE.Vector3(
                    0.2126 * sh[9] + 0.7152 * sh[10] + 0.0722 * sh[11],
                    0.2126 * sh[3] + 0.7152 * sh[4] + 0.0722 * sh[5],
                    0.2126 * sh[6] + 0.7152 * sh[7] + 0.0722 * sh[8],
                );

                if (dir.lengthSq() > 1e-12) {
                    dir.normalize();

                    // 1.023328 = three's irradiance constant for the linear band
                    for (let c = 0; c < 3; c++) {
                        self.k3dEnvLightColor.value.setComponent(c, Math.max(
                            1.023328 * (sh[9 + c] * dir.x + sh[3 + c] * dir.y + sh[6 + c] * dir.z),
                            0.0,
                        ) * envIntensity);
                    }

                    self.k3dEnvLightDir.value.copy(dir.applyMatrix4(rotation)).normalize();
                }
            }

            self.keyLight.visible = false;
            self.headLight.visible = false;
            self.fillLight.visible = false;
            self.backLight.visible = false;

            return;
        }

        for (let b = 0; b < 9; b++) {
            self.k3dEnvSH.value[b].set(0, 0, 0);
        }
        self.k3dEnvLightColor.value.set(0, 0, 0);
        self.k3dEnvRotation.value.identity();
        self.keyLight.visible = value > 0.0;

        if (value <= 1.0) {
            ambientLight.intensity = unlitAmbient
                - (unlitAmbient - initialLightIntensity.ambient) * value;
        } else {
            ambientLight.intensity = initialLightIntensity.ambient;
        }

        self.keyLight.intensity = initialLightIntensity.key * value;
        self.headLight.intensity = initialLightIntensity.head * value;
        self.fillLight.intensity = initialLightIntensity.fill * value;
        self.backLight.intensity = initialLightIntensity.back * value;

        self.backLight.visible = value > 0.0;
        self.headLight.visible = value > 0.0;
        self.fillLight.visible = value > 0.0;
        self.backLight.visible = value > 0.0;
    };

    let pmrem = null;
    let environmentSource = null;
    let environmentEquirect = null;

    // The equirect pole is +Y; scientific data is usually z-up. The user rotation spins
    // the map around the effective up axis.
    function environmentRotation(K3D) {
        const rot = K3D.parameters.environmentRotation || 0.0;

        switch (K3D.parameters.cameraUpAxis) {
            case 'y':
                return new THREE.Euler(0, rot, 0, 'XYZ');
            case 'x':
                return new THREE.Euler(rot, 0, -Math.PI / 2, 'XZY');
            default:
                return new THREE.Euler(Math.PI / 2, 0, rot, 'ZXY');
        }
    }

    self.applyRendererMode = function (K3D) {
        // both PBR modes build the environment - cinematic marches volumes with it;
        // only advanced binds it to the raster materials
        if (K3D.parameters.renderer === 'advanced' || K3D.parameters.renderer === 'cinematic') {
            if (environmentSource !== K3D.parameters.environment || self.scene.environment === null) {
                if (pmrem === null) {
                    pmrem = new THREE.PMREMGenerator(self.renderer);
                }

                // not disposed: the helper hands the same instance to every caller
                if (self.scene.environment) {
                    // pmrem.fromEquirectangular allocates a new render target every time
                    self.scene.environment.dispose();
                }

                environmentEquirect = environmentHelper.getEnvironmentTexture(K3D.parameters.environment);
                self.scene.environment = pmrem.fromEquirectangular(environmentEquirect).texture;
                environmentSource = K3D.parameters.environment;
            }

            const rotation = environmentRotation(K3D);

            self.scene.environmentRotation.copy(rotation);
            self.scene.backgroundRotation.copy(rotation);
            self.scene.background = null;
        } else {
            self.scene.environment = null;
            self.scene.background = null;
        }

        // the mode is switchable from the GUI, so materials compiled for the other one need
        // rebuilding. The AO depth material shares this defines object by reference, so writing
        // the value flips both programs - but only the one told about it recompiles.
        const envLight = K3D.parameters.renderer === 'simple' ? 0 : 1;

        self.K3DObjects.traverse((object) => {
            const { material } = object;

            if (!material || !material.defines
                || typeof (material.defines.K3D_ENV_LIGHT) === 'undefined'
                || material.defines.K3D_ENV_LIGHT === envLight) {
                return;
            }

            material.defines.K3D_ENV_LIGHT = envLight;
            material.needsUpdate = true;

            if (object.userData.k3dAODepthMaterial) {
                object.userData.k3dAODepthMaterial.needsUpdate = true;
            }
        });

        self.recalculateLights(K3D.parameters.lighting);
    };
};
