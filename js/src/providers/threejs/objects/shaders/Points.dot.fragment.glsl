varying vec3 vColor;
uniform float opacity;

#include <clipping_planes_pars_fragment>

void main (void)
{
    #include <clipping_planes_fragment>
    gl_FragColor = vec4(vColor, opacity);
}
