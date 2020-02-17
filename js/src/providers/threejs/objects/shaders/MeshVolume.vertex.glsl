varying vec4 worldPosition;

#include <clipping_planes_pars_vertex>

void main() {
    vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
    worldPosition = modelMatrix * vec4( position, 1.0 );

    #include <clipping_planes_vertex>

    gl_Position = projectionMatrix * mvPosition;
}