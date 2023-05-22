varying vec4 worldPosition;

#include <common>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>

void main() {
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    worldPosition = modelMatrix * vec4(position, 1.0);

    #include <clipping_planes_vertex>

    gl_Position = projectionMatrix * mvPosition;

    gl_Position = projectionMatrix * mvPosition;

    #include <logdepthbuf_vertex>
    #include <clipping_planes_vertex>
}
