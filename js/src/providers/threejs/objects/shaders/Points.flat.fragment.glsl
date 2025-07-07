#include <common>
#include <clipping_planes_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <lights_pars_begin>

uniform float size;
uniform float specular;
uniform float opacity;
uniform mat4 projectionMatrix;

varying vec4 vColor;
varying vec4 mvPosition;

void main(void)
{
    #include <clipping_planes_fragment>

    vec2 impostorSpaceCoordinate = (gl_PointCoord.xy - vec2(0.5, 0.5)) * 2.0;
    float distanceFromCenter = length(impostorSpaceCoordinate);

    if (distanceFromCenter > 1.0) discard;

    float normalizedDepth = sqrt(1.0 - distanceFromCenter * distanceFromCenter);
    float depthOfFragment = normalizedDepth * size * 0.5;

    vec4 pos = vec4(mvPosition.xyz, 1.0);
    pos.z += depthOfFragment;
    pos = projectionMatrix * pos;

    #ifdef USE_LOGDEPTHBUF_EXT
    float depth = log2(1.0 + pos.w) * logDepthBufFC * 0.5;
    #else
    pos = pos / pos.w;
    float depth = ((gl_DepthRange.diff * pos.z) + gl_DepthRange.near + gl_DepthRange.far) / 2.0;
    #endif

    gl_FragDepthEXT = depth;
    float FragCoordZ = pos.z;

    vec4 finalSphereColor = vColor;
    finalSphereColor.a *= opacity;

    gl_FragColor = finalSphereColor;
}
