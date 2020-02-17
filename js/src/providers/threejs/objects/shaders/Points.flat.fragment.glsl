varying vec4 vColor;
uniform float opacity;

#include <clipping_planes_pars_fragment>

void main (void)
{
    #include <clipping_planes_fragment>

    vec2 impostorSpaceCoordinate = (gl_PointCoord.xy - vec2(0.5, 0.5));
    float distanceFromCenter = length(impostorSpaceCoordinate);

    if (distanceFromCenter > 0.5) discard;

    vec4 color = vColor;
    color.a  *= opacity;

    gl_FragColor = color;
}
