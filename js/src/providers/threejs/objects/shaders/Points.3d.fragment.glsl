uniform float size;
uniform float specular;
uniform float opacity;
uniform mat4 projectionMatrix;
uniform vec3 ambientLightColor;

varying vec3 vColor;
varying vec4 mvPosition;

struct DirectionalLight {
    vec3 direction;
    vec3 color;

    int shadow;
    float shadowBias;
    float shadowRadius;
    vec2 shadowMapSize;
};

uniform DirectionalLight directionalLights[NUM_DIR_LIGHTS];

#include <clipping_planes_pars_fragment>

void main (void)
{
    #include <clipping_planes_fragment>

    vec2 impostorSpaceCoordinate = (gl_PointCoord.xy - vec2(0.5, 0.5)) * 2.0;
    float distanceFromCenter = length(impostorSpaceCoordinate);

    if (distanceFromCenter > 1.0) discard;

    float normalizedDepth = sqrt(1.0 - distanceFromCenter * distanceFromCenter);

    #if defined(GL_EXT_frag_depth)
    float depthOfFragment = normalizedDepth * size * 0.5;

    vec4 pos = vec4(mvPosition.xyz, 1.0);
    pos.z += depthOfFragment;

    pos = projectionMatrix * pos;
    pos = pos / pos.w;
    float depth = (pos.z + 1.0) / 2.0;

    if(depth < gl_FragDepthEXT) discard;
    gl_FragDepthEXT = depth;
    #endif

    vec3 normal = vec3(impostorSpaceCoordinate, normalizedDepth);

    vec4 addedLights = vec4(ambientLightColor, 1.0);
    vec4 finalSphereColor = vec4(vColor, opacity);

    for(int l = 0; l <NUM_DIR_LIGHTS; l++) {
        vec3 lightDirection = -directionalLights[l].direction;
        float lightingIntensity = clamp(dot(-lightDirection, normal), 0.0, 1.0);
        addedLights.rgb += directionalLights[l].color * (0.05 + 0.95 * lightingIntensity);

        #if (USE_SPECULAR == 1)
        finalSphereColor.rgb += directionalLights[l].color * pow(lightingIntensity, 80.0);
        #endif
    }

    gl_FragColor = finalSphereColor * addedLights;
}
