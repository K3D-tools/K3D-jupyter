uniform float size;
uniform float specular;

varying vec3 vColor;
varying vec4 mvPosition;
uniform mat4 projectionMatrix;

struct PointLight {
  vec3 color;
  vec3 position;  // light position, in camera coordinates
  float distance; // used for attenuation purposes. Since
                  // we're writing our own shader, it can
                  // really be anything we want (as long as
                  // we assign it to our light in its
                  // "distance" field
};

uniform PointLight pointLights[NUM_POINT_LIGHTS];

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

    vec4 addedLights = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 finalSphereColor = vec4(vColor, 1.0);

    for(int l = 0; l <NUM_POINT_LIGHTS; l++) {
        vec3 lightDirection = normalize(vec3(mvPosition) - pointLights[l].position);
        float lightingIntensity = clamp(dot(-lightDirection, normal), 0.0, 1.0);
        addedLights.rgb += pointLights[l].color * (0.05 + 0.95 * lightingIntensity);

        #if (USE_SPECULAR == 1)
        finalSphereColor.rgb += pointLights[l].color * pow(lightingIntensity, 80.0);
        #endif
    }

    gl_FragColor = finalSphereColor * addedLights;
}
