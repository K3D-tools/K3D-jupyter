varying vec3 vColor;

#include <clipping_planes_pars_fragment>

void main (void)
{
    #include <clipping_planes_fragment>

    vec2 impostorSpaceCoordinate = (gl_PointCoord.xy - vec2(0.5, 0.5));
    float distanceFromCenter = length(impostorSpaceCoordinate);

    if (distanceFromCenter > 0.5) discard;

    gl_FragColor = vec4(vColor, 1.0);
}
