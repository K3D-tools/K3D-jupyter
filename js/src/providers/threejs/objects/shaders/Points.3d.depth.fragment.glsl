#include <common>
#include <clipping_planes_pars_fragment>
#include <logdepthbuf_pars_fragment>

uniform float size;
uniform mat4 projectionMatrix;

varying vec4 vColor;
varying vec4 mvPosition;

#if (USE_PER_POINT_SIZE == 1)
varying float vPointSize;
#endif

// The AO depth prepass: the override material would rasterise the billboard quads,
// this writes the analytic sphere depth instead - same convention as the peel depth
// pass (raw depth in .r, background cleared to 1.0).
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

    // GTAO reconstructs view positions through the inverse projection, so the colour
    // channel always carries the linear window-space depth - the mesh prepass writes
    // gl_FragCoord.z there. Only the depth test follows the renderer's buffer encoding.
    float linearDepth = ((gl_DepthRange.diff * (pos.z / pos.w))
        + gl_DepthRange.near + gl_DepthRange.far) / 2.0;

    #ifdef USE_LOGARITHMIC_DEPTH_BUFFER
    gl_FragDepthEXT = log2(1.0 + pos.w) * logDepthBufFC * 0.5;
    #else
    gl_FragDepthEXT = linearDepth;
    #endif

    gl_FragColor = vec4(linearDepth, 0.0, 0.0, 1.0);
}
