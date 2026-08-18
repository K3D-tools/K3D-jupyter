#include <common>
#include <clipping_planes_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <lights_pars_begin>

// minimal mirror of the fields BRDF_GGX reads - the full struct lives in
// lights_physical_pars_fragment, which drags in the whole mesh lighting pipeline
struct PhysicalMaterial {
    vec3 diffuseColor;
    float roughness;
    vec3 specularColorBlended;
    float specularF90;
};

// K3D_GGX_CHUNK

uniform vec3 k3dEnvSH[9];
uniform mat3 k3dEnvRotation;
uniform vec3 k3dEnvLightDir;
uniform vec3 k3dEnvLightColor;
uniform float k3dEnvSurfaceBoost;
uniform float size;
uniform float opacity;
uniform float roughness;
uniform float metalness;
uniform mat4 projectionMatrix;

varying vec4 vColor;
varying vec4 mvPosition;

#if (USE_PER_POINT_SIZE == 1)
varying float vPointSize;
#endif

void main(void)
{
    #include <clipping_planes_fragment>

    vec2 impostorSpaceCoordinate = (gl_PointCoord.xy - vec2(0.5, 0.5)) * 2.0;
    float distanceFromCenter = length(impostorSpaceCoordinate);

    if (distanceFromCenter > 1.0) discard;

    float normalizedDepth = sqrt(1.0 - distanceFromCenter * distanceFromCenter);

    #if (USE_PER_POINT_SIZE == 1)
    float depthOfFragment = normalizedDepth * vPointSize * 0.5;
    #else
    float depthOfFragment = normalizedDepth * size * 0.5;
    #endif

    vec4 pos = vec4(mvPosition.xyz, 1.0);
    pos.z += depthOfFragment;
    pos = projectionMatrix * pos;

    #ifdef USE_LOGARITHMIC_DEPTH_BUFFER
    float depth = log2(1.0 + pos.w) * logDepthBufFC * 0.5;
    #else
    pos = pos / pos.w;
    float depth = ((gl_DepthRange.diff * pos.z) + gl_DepthRange.near + gl_DepthRange.far) / 2.0;
    #endif

    gl_FragDepthEXT = depth;
    float fragCoordZ = pos.z;

    // everything below is view space: the analytic sphere normal, the surface point
    // and three's directional light directions all live there already
    vec3 normal = vec3(impostorSpaceCoordinate, normalizedDepth);
    vec3 viewDir = normalize(-vec3(mvPosition.xy, mvPosition.z + depthOfFragment));
    vec3 kWorldNormal = normalize(normal * mat3(viewMatrix));

    vec4 finalSphereColor = vColor;
    finalSphereColor.a *= opacity;

    PhysicalMaterial material;
    material.diffuseColor = finalSphereColor.rgb * (1.0 - metalness);
    material.roughness = max(roughness, 0.0525);
    material.specularColorBlended = mix(vec3(0.04), finalSphereColor.rgb, metalness);
    material.specularF90 = 1.0;

    // rig ambient + environment SH irradiance; the impostor is a surface, so the env
    // part gets the same delivery correction PMREM materials take from environmentIntensity
    vec3 irradiance = ambientLightColor
        + shGetIrradianceAt(k3dEnvRotation * kWorldNormal, k3dEnvSH) * k3dEnvSurfaceBoost;
    vec3 diffuse = irradiance * BRDF_Lambert(material.diffuseColor);
    vec3 specular = vec3(0.0);

    #if NUM_DIR_LIGHTS > 0
    for (int l = 0; l < NUM_DIR_LIGHTS; l++) {
        vec3 lightDir = directionalLights[l].direction;
        vec3 lightIrradiance = directionalLights[l].color * clamp(dot(lightDir, normal), 0.0, 1.0);

        diffuse += lightIrradiance * BRDF_Lambert(material.diffuseColor);
        specular += lightIrradiance * BRDF_GGX(lightDir, viewDir, normal, material);
    }
    #endif

    // advanced: the dominant directional light distilled from the environment's L1 band
    {
        vec3 lightDir = normalize((viewMatrix * vec4(k3dEnvLightDir, 0.0)).xyz);
        vec3 lightIrradiance = k3dEnvLightColor * k3dEnvSurfaceBoost
            * clamp(dot(lightDir, normal), 0.0, 1.0);

        diffuse += lightIrradiance * BRDF_Lambert(material.diffuseColor);
        specular += lightIrradiance * BRDF_GGX(lightDir, viewDir, normal, material);
    }

    gl_FragColor = vec4(diffuse + specular, finalSphereColor.a);
}
