#include <common>
#include <clipping_planes_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <normal_pars_vertex>

varying vec3 localPosition;
varying vec3 vViewPosition;

void main() {
    #include <beginnormal_vertex>

    #include <begin_vertex>
    #include <project_vertex>
    #include <defaultnormal_vertex>
    #include <normal_vertex>

    localPosition = position + vec3(0.5, 0.5, 0.5);
    vViewPosition = -mvPosition.xyz;

    #include <worldpos_vertex>
    #include <logdepthbuf_vertex>
    #include <clipping_planes_vertex>

}
