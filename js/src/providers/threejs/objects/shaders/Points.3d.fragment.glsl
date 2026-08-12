#include <common>
#include <clipping_planes_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <lights_pars_begin>

uniform float size;
uniform float opacity;
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

    vec3 normal = vec3(impostorSpaceCoordinate, normalizedDepth);

    vec4 addedLights = vec4(ambientLightColor * RECIPROCAL_PI, 1.0);
    vec4 finalSphereColor = vColor;
    vec3 specularColor = vec3(0.0);

    finalSphereColor.a *= opacity;

    for (int l = 0; l < NUM_DIR_LIGHTS; l++) {
        vec3 lightDirection = -directionalLights[l].direction;
        vec3 lightColor = directionalLights[l].color * RECIPROCAL_PI;
        float lightingIntensity = clamp(dot(-lightDirection, normal), 0.0, 1.0);
        addedLights.rgb += lightColor * (0.05 + 0.95 * lightingIntensity);

        #if (USE_SPECULAR == 1)
        specularColor += lightColor * pow(lightingIntensity, 80.0);
        #endif
    }

    gl_FragColor = finalSphereColor * addedLights;
    gl_FragColor.rgb += specularColor;
}
