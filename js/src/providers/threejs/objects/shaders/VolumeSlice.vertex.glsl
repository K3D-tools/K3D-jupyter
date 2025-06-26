#include <common>
#include <clipping_planes_pars_vertex>
#include <logdepthbuf_pars_vertex>

attribute vec3 normals;
uniform vec3 volumeSize[TEXTURE_COUNT];
varying vec3 coord;

void main() {
    coord = position + vec3(0.5) + 0.5 / volumeSize[0];

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

    #include <clipping_planes_vertex>

    gl_Position = projectionMatrix * mvPosition;

    #include <logdepthbuf_vertex>
}