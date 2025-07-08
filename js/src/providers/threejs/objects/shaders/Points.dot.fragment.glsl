varying vec4 vColor;
uniform float opacity;

#include <common>
#include <clipping_planes_pars_fragment>
#include <logdepthbuf_pars_fragment>

varying vec4 mvPosition;

void main(void)
{
    #include <clipping_planes_fragment>
    #include <logdepthbuf_fragment>

    vec4 color = vColor;
    color.a *= opacity;

    float fragCoordZ = mvPosition.z;

    gl_FragColor = color;
}
