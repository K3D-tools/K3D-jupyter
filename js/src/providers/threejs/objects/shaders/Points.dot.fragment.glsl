varying vec4 vColor;
uniform float opacity;

#include <common>
#include <clipping_planes_pars_fragment>
#include <logdepthbuf_pars_fragment>

void main (void)
{
    #include <clipping_planes_fragment>
    #include <logdepthbuf_fragment>

    vec4 color = vColor;
    color.a  *= opacity;

    gl_FragColor = color;
}
