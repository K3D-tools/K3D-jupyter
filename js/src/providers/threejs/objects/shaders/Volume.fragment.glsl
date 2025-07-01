#include <common>
#include <clipping_planes_pars_fragment>
#include <lights_pars_begin>

precision highp sampler3D;

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
uniform float samples;
uniform float alpha_coef;
uniform float gradient_step;

uniform vec4 scale;
uniform vec4 translation;

uniform sampler3D mask;
uniform float maskOpacities[256];

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
    return normalize(
        vec3(
            px - getMaskedVolume(pos + vec3(gradient_step, 0, 0)),
            px - getMaskedVolume(pos + vec3(0, gradient_step, 0)),
            px - getMaskedVolume(pos + vec3(0, 0, gradient_step))
        )
    );
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

        vec3 textcoord_end = ((eye + direction * tmax) - translation.xyz) / scale.xyz + vec3(0.5);
        #else
        vec4 value = vec4(0.0, 0.0, 0.0, 0.0);
        vec3 direction = normalize(transformedWorldPosition - transformedCameraPosition);
        intersect(makeRay(transformedCameraPosition, direction), aabb, tmin, tmax);

        vec3 textcoord_end = localPosition + vec3(0.5);
        #endif
        vec3 textcoord_start = textcoord_end - (tmax - max(0.0, tmin)) * direction / scale.xyz;
        vec3 textcoord_delta = textcoord_end - textcoord_start;

        int sampleCount = min(int(length(textcoord_delta) * samples), int(samples * 1.8));

        textcoord_delta = textcoord_delta / float(sampleCount);
        textcoord_start = textcoord_start - textcoord_delta * (0.01 + 0.98 * jitter);

        vec3 textcoord = textcoord_start - textcoord_delta;

        float step = length(textcoord_delta);

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

                    pxColor.rgb *= pxColor.a;

                    // LIGHT
                    #if NUM_DIR_LIGHTS > 0
                    if (pxColor.a > 0.0) {
                        vec4 addedLights = vec4(ambientLightColor / PI, 1.0);
                        vec3 specularColor = vec3(0.0);

                        vec3 normal = worldGetNormal(px * maskOpacity, textcoord);

                        vec3 lightDirection;
                        float lightingIntensity;

                        vec3 lightReflect;
                        float specularFactor;

                        #pragma unroll_loop_start
                        for (int i = 0; i < NUM_DIR_LIGHTS; i++) {
                            lightDirection = directionalLights[i].direction;
                            lightingIntensity = clamp(dot(lightDirection, normal), 0.0, 1.0);
                            addedLights.rgb += directionalLights[i].color / PI * (0.2 + 0.8 * lightingIntensity) * (1.0 - shadow);

                            lightReflect = normalize(reflect(lightDirection, normal));
                            specularFactor = dot(direction, lightReflect);

                            if (specularFactor > 0.0)
                            specularColor += 0.002 * scaled_px * (1.0 / step) *
                            directionalLights[i].color / PI * pow(specularFactor, 250.0) *
                            pxColor.a * (1.0 - shadow);
                        }
                        #pragma unroll_loop_end

                        pxColor.rgb = pxColor.rgb * addedLights.xyz + specularColor;
                    }
                    #endif

                    value += pxColor;

                    if (value.a >= 0.99) {
                        value.a = 1.0;
                        break;
                    }
                }
            }
        }

        #if (RAY_SAMPLES_COUNT > 0)

        accuColor += value;
    }

    gl_FragColor = accuColor / float(RAY_SAMPLES_COUNT);

    #else

    gl_FragColor = value;

    #endif
}
