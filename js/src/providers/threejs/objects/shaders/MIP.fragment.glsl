#include <common>
#include <clipping_planes_pars_fragment>
#include <lights_pars_begin>

// minimal mirror of the fields BRDF_GGX reads (the full struct lives in
// lights_physical_pars_fragment, which assumes the mesh pipeline)
struct PhysicalMaterial {
    vec3 diffuseColor;
    float roughness;
    vec3 specularColorBlended;
    float specularF90;
};

// K3D_GGX_CHUNK

precision highp sampler3D;

uniform vec3 k3dEnvSH[9];
uniform mat3 k3dEnvRotation;
uniform vec3 k3dEnvLightDir;
uniform vec3 k3dEnvLightColor;
uniform mat4 transform;
uniform sampler3D volumeTexture;

uniform sampler2D colormap;
uniform sampler2D jitterTexture;
uniform float low;
uniform float high;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform float samples;
uniform float gradient_step;
uniform float roughness;
uniform float metalness;

uniform sampler3D mask;
uniform float maskOpacities[256];

uniform vec4 scale;
uniform vec4 translation;

varying vec3 localPosition;
varying vec3 transformedCameraPosition;
varying vec3 transformedWorldPosition;

float inv_range;

struct Ray {
    vec3 origin;
    vec3 direction;
    vec3 inv_direction;
    int sign[3];
};

vec3 aabb[2] = vec3[2](
    vec3(-0.5, -0.5, -0.5),
    vec3(0.5, 0.5, 0.5)
);

Ray makeRay(vec3 origin, vec3 direction) {
    vec3 inv_direction = vec3(1.0) / direction;

    return Ray(
        origin,
        direction,
        inv_direction,
        int[3](
            ((inv_direction.x < 0.0) ? 1 : 0),
            ((inv_direction.y < 0.0) ? 1 : 0),
            ((inv_direction.z < 0.0) ? 1 : 0)
        )
    );
}

/*
	From: https://github.com/hpicgs/cgsee/wiki/Ray-Box-Intersection-on-the-GPU
*/
void intersect(
in Ray ray, in vec3 aabb[2],
out float tmin, out float tmax
) {
    float tymin, tymax, tzmin, tzmax;
    tmin = (aabb[ray.sign[0]].x - ray.origin.x) * ray.inv_direction.x;
    tmax = (aabb[1 - ray.sign[0]].x - ray.origin.x) * ray.inv_direction.x;
    tymin = (aabb[ray.sign[1]].y - ray.origin.y) * ray.inv_direction.y;
    tymax = (aabb[1 - ray.sign[1]].y - ray.origin.y) * ray.inv_direction.y;
    tzmin = (aabb[ray.sign[2]].z - ray.origin.z) * ray.inv_direction.z;
    tzmax = (aabb[1 - ray.sign[2]].z - ray.origin.z) * ray.inv_direction.z;
    tmin = max(max(tmin, tymin), tzmin);
    tmax = min(min(tmax, tymax), tzmax);
}

float getMaskOpacity(vec3 pos) {
    int maskValue = int(texture(mask, pos).r * 255.0);

    return maskOpacities[maskValue];
}

float getMaskedVolume(vec3 pos)
{
    #if (USE_MASK == 1)
    return texture(volumeTexture, pos).x * getMaskOpacity(pos);
    #else
    return texture(volumeTexture, pos).x;
    #endif
}

vec3 worldGetNormal(in float px, in vec3 pos)
{
    vec3 gradient = vec3(
        px - getMaskedVolume(pos + vec3(gradient_step, 0, 0)),
        px - getMaskedVolume(pos + vec3(0, gradient_step, 0)),
        px - getMaskedVolume(pos + vec3(0, 0, gradient_step))
    );

    // saturated plateaus have no gradient: normalize(0) is NaN and one NaN sample
    // blacks out the ray (0 * NaN stays NaN even with zeroed SH)
    if (dot(gradient, gradient) < 1e-20) {
        return vec3(0.0);
    }

    return normalize(gradient);
}

void main() {
    float jitter = texture2D(jitterTexture, gl_FragCoord.xy / 64.0).r;
    float tmin = 0.0;
    float tmax = 0.0;
    float px = -3.402823466e+38F;
    vec4 pxColor = vec4(0.0, 0.0, 0.0, 0.0);

    inv_range = 1.0 / (high - low);
    aabb[0] = aabb[0] * scale.xyz + translation.xyz;
    aabb[1] = aabb[1] * scale.xyz + translation.xyz;

    vec3 direction = normalize(transformedWorldPosition - transformedCameraPosition);
    intersect(makeRay(transformedCameraPosition, direction), aabb, tmin, tmax);

    vec3 textcoord_end = localPosition + vec3(0.5);
    vec3 textcoord_start = textcoord_end - (tmax - max(0.0, tmin)) * direction / scale.xyz;
    vec3 textcoord_delta = textcoord_end - textcoord_start;

    int sampleCount = min(int(length(textcoord_delta) * samples), int(samples * 1.8));

    textcoord_delta = textcoord_delta / float(sampleCount);
    #ifdef K3D_AO_DEPTH_PASS
    // no jitter: a per-pixel noisy shell depth reads as micro-cliffs to GTAO
    textcoord_start = textcoord_start - textcoord_delta * 0.5;
    #else
    textcoord_start = textcoord_start - textcoord_delta * (0.01 + 0.98 * jitter);
    #endif

    vec3 textcoord = textcoord_start - textcoord_delta;
    vec3 maxTextcoord = textcoord;

    float step = length(textcoord_delta);

    for (int count = 0; count < sampleCount; count++) {
        textcoord += textcoord_delta;

        #if NUM_CLIPPING_PLANES > 0
        vec4 plane;
        vec3 pos = -vec3(modelViewMatrix * vec4(textcoord - vec3(0.5), 1.0));

        #pragma unroll_loop_start
        for (int i = 0; i < UNION_CLIPPING_PLANES; i++) {
            plane = clippingPlanes[i];
            if (dot(pos, plane.xyz) > plane.w) continue;
        }
        #pragma unroll_loop_end
        #endif

        #if (USE_MASK == 1)
        float newPx = texture(volumeTexture, textcoord).x * getMaskOpacity(textcoord);
        #else
        float newPx = texture(volumeTexture, textcoord).x;
        #endif

        if (newPx > px) {
            px = newPx;
            maxTextcoord = textcoord;
        }

        if (px >= high) {
            break;
        }
    }

    float scaled_px = (px - low) * inv_range;

    if (scaled_px > 0.0) {
        scaled_px = min(scaled_px, 0.99);

        pxColor = texture(colormap, vec2(scaled_px, 0.5));
    }

    #ifdef K3D_AO_DEPTH_PASS
    // the occluder shell: depth of the maximum-intensity point, when opaque enough
    if (pxColor.a >= 0.5) {
        vec4 kClipPos = projectionMatrix * modelViewMatrix * vec4(maxTextcoord - vec3(0.5), 1.0);
        float kShellDepth = ((gl_DepthRange.diff * (kClipPos.z / kClipPos.w))
            + gl_DepthRange.near + gl_DepthRange.far) / 2.0;

        gl_FragDepthEXT = kShellDepth;
        // g == 2.0 marks a volumetric shell - the AO overlay halves occlusion
        // there (mesh depth packing keeps g below 1.0)
        gl_FragColor = vec4(kShellDepth, 2.0, 0.0, 1.0);
        return;
    }
    discard;
    #endif

    // LIGHT
    vec3 normal = worldGetNormal(px, maxTextcoord);
    vec4 addedLights = vec4(
        (ambientLightColor + shGetIrradianceAt(k3dEnvRotation * normal, k3dEnvSH)) * RECIPROCAL_PI, 1.0);
    vec3 specularColor = vec3(0.0);

    PhysicalMaterial specMaterial;
    specMaterial.diffuseColor = vec3(0.0);
    specMaterial.roughness = max(roughness, 0.0525);
    specMaterial.specularColorBlended = mix(vec3(0.04), pxColor.rgb, metalness);
    specMaterial.specularF90 = 1.0;

    #if NUM_DIR_LIGHTS > 0
    vec3 lightDirection;
    vec3 lightColor;
    float lightingIntensity;

    #pragma unroll_loop_start
    for (int i = 0; i < NUM_DIR_LIGHTS; i++) {
        lightDirection = -directionalLights[i].direction;
        lightColor = directionalLights[i].color * RECIPROCAL_PI;
        lightingIntensity = clamp(dot(-lightDirection, normal), 0.0, 1.0);
        addedLights.rgb += lightColor * (0.05 + 0.95 * lightingIntensity);

        #if (USE_SPECULAR == 1)
        specularColor += lightColor * lightingIntensity *
        BRDF_GGX(-lightDirection, -direction, normal, specMaterial) * pxColor.a;
        #endif
    }
    #pragma unroll_loop_end
    #endif

    // advanced: the dominant directional light distilled from the environment's L1 band
    {
        vec3 envLightColor = k3dEnvLightColor * RECIPROCAL_PI;
        float envIntensity = clamp(dot(k3dEnvLightDir, normal), 0.0, 1.0);
        addedLights.rgb += envLightColor * (0.05 + 0.95 * envIntensity);

        #if (USE_SPECULAR == 1)
        specularColor += envLightColor * envIntensity *
        BRDF_GGX(k3dEnvLightDir, -direction, normal, specMaterial) * pxColor.a;
        #endif
    }

    // no (1 - metalness) on the body - same reasoning as in Volume.fragment.glsl:
    // metalness tints and strengthens the highlights instead of going black
    pxColor.rgb *= addedLights.xyz;
    pxColor.rgb += specularColor;

    gl_FragColor = pxColor;
}
