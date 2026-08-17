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
uniform vec3 lightMapSize;
uniform vec2 lightMapRenderTargetSize;
uniform sampler2D shadowTexture;

uniform mat4 transform;
uniform sampler3D volumeTexture;

uniform sampler2D colormap;
uniform sampler2D jitterTexture;
uniform float focal_length;
uniform float focal_plane;
uniform float low;
uniform float high;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform float samples;
uniform float alpha_coef;
uniform float gradient_step;
uniform float roughness;
uniform float metalness;

uniform vec4 scale;
uniform vec4 translation;
uniform vec4 rotation;

// depth-peel segment bounds (issue #277): with uPeelSegment == 1 the march is
// clamped between two peel-layer depth textures, so meshes interleave correctly
uniform int uPeelSegment;
uniform sampler2D uPeelNearTexture;
uniform sampler2D uPeelFarTexture;
uniform vec2 uPeelSize;
uniform mat4 uPeelInvProjection;
uniform mat4 uPeelInvView;

uniform sampler3D mask;
uniform float maskOpacities[256];

vec3 rotate_vertex_position(vec3 pos, vec3 t, vec4 q) {
    vec3 p = pos.xyz - t.xyz;

    return p.xyz + 2.0 * cross(cross(p.xyz, q.xyz) + q.w * p.xyz, q.xyz) + t.xyz;
}

// window-space z from a peel layer -> distance along the ray in the marching space
// (the quaternion rotation is an isometry, so t carries over from world unchanged)
float peelT(sampler2D depthTexture, vec3 origin, vec3 dir, float noHitT) {
    vec2 uv = gl_FragCoord.xy * uPeelSize;
    float z = texture2D(depthTexture, uv).r;

    if (z >= 1.0) {
        return noHitT;
    }

    vec4 view = uPeelInvProjection * vec4(uv * 2.0 - 1.0, z * 2.0 - 1.0, 1.0);
    view /= view.w;

    vec4 world = uPeelInvView * view;
    vec3 p = rotate_vertex_position(world.xyz, translation.xyz, rotation);

    return dot(p - origin, dir);
}

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
    // blacks out the whole ray (0 * NaN stays NaN even with zeroed SH)
    if (dot(gradient, gradient) < 1e-20) {
        return vec3(0.0);
    }

    return normalize(gradient);
}

float getShadow(vec3 textcoord, vec2 sliceCount)
{
    float zidx1 = floor(textcoord.z * lightMapSize.z);
    float zidx2 = ceil(textcoord.z * lightMapSize.z);

    float shadow1 = texture2D(shadowTexture,
                              vec2(
                                  floor(mod(zidx1, sliceCount.x)) * lightMapSize.x / lightMapRenderTargetSize.x,
                                  floor(zidx1 / sliceCount.x) * lightMapSize.y / lightMapRenderTargetSize.y
                              )
                              + vec2(textcoord.x / sliceCount.x, textcoord.y / sliceCount.y)
    ).r;

    float shadow2 = texture2D(shadowTexture,
                              vec2(
                                  floor(mod(zidx2, sliceCount.x)) * lightMapSize.x / lightMapRenderTargetSize.x,
                                  floor(zidx2 / sliceCount.x) * lightMapSize.y / lightMapRenderTargetSize.y
                              )
                              + vec2(textcoord.x / sliceCount.x, textcoord.y / sliceCount.y)
    ).r;

    return mix(shadow1, shadow2, textcoord.z * lightMapSize.z - zidx1);
}

void main() {
    float jitter = texture2D(jitterTexture, gl_FragCoord.xy / 64.0).r;
    float tmin = 0.0;
    float tmax = 0.0;
    float px = 0.0;
    float shadow = 0.0;
    vec4 pxColor = vec4(0.0, 0.0, 0.0, 0.0);

    inv_range = 1.0 / (high - low);
    aabb[0] = aabb[0] * scale.xyz + translation.xyz;
    aabb[1] = aabb[1] * scale.xyz + translation.xyz;

    #if (RAY_SAMPLES_COUNT > 0)
    vec4 accuColor = vec4(0.0, 0.0, 0.0, 0.0);

    for (int ray_samples = 0; ray_samples < RAY_SAMPLES_COUNT; ray_samples++) {

        vec4 value = vec4(0.0, 0.0, 0.0, 0.0);

        vec3 direction = normalize(transformedWorldPosition - transformedCameraPosition);

        // Focal plane correction
        vec3 P = transformedCameraPosition + direction * focal_plane;

        float r = texture2D(jitterTexture, vec2(0.3) + gl_FragCoord.xy / 64.0 * float(ray_samples + 3)).r;
        vec3 apertureShift = normalize(vec3(
                                           1.0 - 2.0 * texture2D(jitterTexture, vec2(0.0) + gl_FragCoord.xy / 64.0 * float(ray_samples)).r,
                                           1.0 - 2.0 * texture2D(jitterTexture, vec2(0.1) + gl_FragCoord.xy / 64.0 * float(ray_samples + 1)).r,
                                           1.0 - 2.0 * texture2D(jitterTexture, vec2(0.2) + gl_FragCoord.xy / 64.0 * float(ray_samples + 2)).r
                                       )) * r * focal_length;

        direction = normalize(P - (transformedCameraPosition + apertureShift));

        vec3 eye = P - direction * 1000000.0;

        intersect(makeRay(eye, direction), aabb, tmin, tmax);

        vec3 rayOrigin = eye;
        #else
        vec4 value = vec4(0.0, 0.0, 0.0, 0.0);
        vec3 direction = normalize(transformedWorldPosition - transformedCameraPosition);
        intersect(makeRay(transformedCameraPosition, direction), aabb, tmin, tmax);

        vec3 rayOrigin = transformedCameraPosition;
        #endif
        // the sampling grid is anchored to the whole box, never to the segment -
        // every segmentation samples identical positions, so layer joints cannot seam
        float tBox = max(0.0, tmin);
        vec3 gridStart = ((rayOrigin + direction * tBox) - translation.xyz) / scale.xyz + vec3(0.5);
        vec3 gridSpan = ((rayOrigin + direction * tmax) - translation.xyz) / scale.xyz + vec3(0.5) - gridStart;

        int totalCount = max(min(int(length(gridSpan) * samples), int(samples * 1.8)), 1);
        vec3 textcoord_delta = gridSpan / float(totalCount);
        float tStep = (tmax - tBox) / float(totalCount);

        #ifdef K3D_AO_DEPTH_PASS
        // no jitter: a per-pixel noisy shell depth reads as micro-cliffs to GTAO
        float jitterOffset = 0.5;
        #else
        float jitterOffset = 0.01 + 0.98 * jitter;
        #endif

        int kMin = 0;
        int kMax = totalCount - 1;

        #ifndef K3D_AO_DEPTH_PASS
        if (uPeelSegment == 1) {
            // sample k sits at t = tBox + (k - jitterOffset) * tStep; a boundary sample
            // belongs to the next segment (>= near, < far), so nothing is counted twice
            float tNear = peelT(uPeelNearTexture, rayOrigin, direction, -1.0);
            float tFar = peelT(uPeelFarTexture, rayOrigin, direction, -1.0);

            if (tNear < 0.0) {
                // peeling only ever leaves deeper layers where the nearer one exists -
                // an empty near layer means an earlier segment already reached the exit
                kMax = -1;
            } else {
                kMin = max(0, int(ceil((min(tNear, tmax) - tBox) / tStep + jitterOffset)));
            }

            if (tFar >= 0.0) {
                kMax = min(kMax, int(ceil((min(tFar, tmax) - tBox) / tStep + jitterOffset)) - 1);
            }
        }
        #endif

        int sampleCount = max(kMax - kMin + 1, 0);
        vec3 textcoord_start = gridStart + (float(kMin) - jitterOffset) * textcoord_delta;

        vec3 textcoord = textcoord_start - textcoord_delta;

        float step = length(textcoord_delta);

        #ifdef K3D_AO_DEPTH_PASS
        float kPrevAlpha = 0.0;
        #endif

        #if (USE_SHADOW == 1)
        float sliceSize = lightMapSize.x * lightMapSize.y;
        vec2 sliceCount = lightMapRenderTargetSize / lightMapSize.xy;
        #endif

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

            px = texture(volumeTexture, textcoord).x;
            float scaled_px = (px - low) * inv_range;

            if (scaled_px > 0.0) {
                #if (USE_MASK == 1)
                float maskOpacity = getMaskOpacity(textcoord);
                #else
                float maskOpacity = 1.0;
                #endif

                if (maskOpacity > 0.0) {
                    #if (USE_SHADOW == 1)
                    shadow =
                    (getShadow(textcoord, sliceCount) +
                    getShadow(textcoord + vec3(1.0 / lightMapSize.x, 0, 0), sliceCount) +
                    getShadow(textcoord - vec3(1.0 / lightMapSize.x, 0, 0), sliceCount) +
                    getShadow(textcoord + vec3(0, 1.0 / lightMapSize.y, 0), sliceCount) +
                    getShadow(textcoord - vec3(0, 1.0 / lightMapSize.y, 0), sliceCount) +
                    getShadow(textcoord + vec3(0, 0, 1.0 / lightMapSize.z), sliceCount) +
                    getShadow(textcoord - vec3(0, 0, 1.0 / lightMapSize.z), sliceCount)) / 7.0;
                    #else
                    shadow = 0.0;
                    #endif

                    scaled_px = min(scaled_px, 0.99);

                    pxColor = texture(colormap, vec2(scaled_px, 0.5));

                    pxColor.a = 1.0 - pow(1.0 - pxColor.a, step * alpha_coef);
                    pxColor.a *= (1.0 - value.a);
                    pxColor.a *= maskOpacity;

                    // straight colormap colour, kept for the metal tint before the
                    // premultiply darkens rgb by alpha
                    vec3 kBaseColor = pxColor.rgb;

                    pxColor.rgb *= pxColor.a;

                    // LIGHT (skipped in the AO depth pass - only opacity matters there)
                    #ifndef K3D_AO_DEPTH_PASS
                    if (pxColor.a > 0.0) {
                        vec3 normal = worldGetNormal(px * maskOpacity, textcoord);
                        vec4 addedLights = vec4(
                            (ambientLightColor + shGetIrradianceAt(k3dEnvRotation * normal, k3dEnvSH)) * RECIPROCAL_PI, 1.0);
                        vec3 specularColor = vec3(0.0);

                        PhysicalMaterial specMaterial;
                        specMaterial.diffuseColor = vec3(0.0);
                        specMaterial.roughness = max(roughness, 0.0525);
                        specMaterial.specularColorBlended = mix(vec3(0.04), kBaseColor, metalness);
                        specMaterial.specularF90 = 1.0;

                        #if NUM_DIR_LIGHTS > 0
                        vec3 lightDirection;
                        vec3 lightColor;
                        float lightingIntensity;

                        #pragma unroll_loop_start
                        for (int i = 0; i < NUM_DIR_LIGHTS; i++) {
                            lightDirection = directionalLights[i].direction;
                            lightColor = directionalLights[i].color * RECIPROCAL_PI;
                            lightingIntensity = clamp(dot(lightDirection, normal), 0.0, 1.0);
                            addedLights.rgb += lightColor * (0.2 + 0.8 * lightingIntensity) * (1.0 - shadow);

                            specularColor += 0.01 * scaled_px * (1.0 / step) *
                            lightColor * lightingIntensity *
                            BRDF_GGX(lightDirection, -direction, normal, specMaterial) *
                            pxColor.a * (1.0 - shadow);
                        }
                        #pragma unroll_loop_end
                        #endif

                        // advanced: the dominant directional light distilled from the
                        // environment's L1 band (zero in simple)
                        {
                            vec3 envLightColor = k3dEnvLightColor * RECIPROCAL_PI;
                            float envIntensity = clamp(dot(k3dEnvLightDir, normal), 0.0, 1.0);
                            addedLights.rgb += envLightColor * (0.2 + 0.8 * envIntensity) * (1.0 - shadow);

                            specularColor += 0.01 * scaled_px * (1.0 / step) *
                            envLightColor * envIntensity *
                            BRDF_GGX(k3dEnvLightDir, -direction, normal, specMaterial) *
                            pxColor.a * (1.0 - shadow);
                        }

                        // no (1 - metalness) on the body: a volume cannot sample the
                        // environment specularly, and with F0 == base colour the metal
                        // ambient response equals the diffuse one anyway. Metalness
                        // tints and strengthens the highlights instead of going black.
                        pxColor.rgb = pxColor.rgb * addedLights.xyz + specularColor;
                    }
                    #endif

                    value += pxColor;

                    #ifdef K3D_AO_DEPTH_PASS
                    if (value.a >= 0.5) {
                        // the occluder shell: depth of the point where accumulated
                        // opacity crosses one half. Sub-step interpolation: a depth
                        // quantised to the march step reads as per-pixel cliffs to GTAO
                        float kT = (0.5 - kPrevAlpha) / max(value.a - kPrevAlpha, 1e-6);
                        vec3 kShellPos = textcoord - textcoord_delta * (1.0 - clamp(kT, 0.0, 1.0));
                        vec4 kClipPos = projectionMatrix * modelViewMatrix
                            * vec4(kShellPos - vec3(0.5), 1.0);
                        float kShellDepth = ((gl_DepthRange.diff * (kClipPos.z / kClipPos.w))
                            + gl_DepthRange.near + gl_DepthRange.far) / 2.0;

                        gl_FragDepthEXT = kShellDepth;
                        // g == 2.0 marks a volumetric shell - the AO overlay halves
                        // occlusion there (mesh depth packing keeps g below 1.0)
                        gl_FragColor = vec4(kShellDepth, 2.0, 0.0, 1.0);
                        return;
                    }
                    kPrevAlpha = value.a;
                    #endif

                    if (value.a >= 0.99) {
                        value.a = 1.0;
                        break;
                    }
                }
            }
        }

        #ifdef K3D_AO_DEPTH_PASS
        // no crossing: still a volume-composited pixel. Marked at (almost) the far
        // plane: the overlay classifies it as volumetric - a discard left faint
        // regions in the mesh AO class, and a nearby mesh printed its GTAO onto the
        // ray integral as dark blotches. Far depth adds no occluder geometry, and
        // anything real inside or behind the box still wins the depth test.
        gl_FragDepthEXT = 0.999999;
        gl_FragColor = vec4(0.999999, 2.0, 0.0, 1.0);
        return;
        #endif

        #if (RAY_SAMPLES_COUNT > 0)

        accuColor += value;
    }

    gl_FragColor = accuColor / float(RAY_SAMPLES_COUNT);

    #else

    gl_FragColor = value;

    #endif
}
