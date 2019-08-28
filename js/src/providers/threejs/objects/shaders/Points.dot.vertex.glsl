uniform float size;
uniform float scale;

attribute vec3 color;

varying vec3 vColor;
varying vec4 mvPosition;

#include <clipping_planes_pars_vertex>

void main() {
    mvPosition = modelViewMatrix * vec4( position, 1.0 );
    gl_PointSize = 2.0 * size;

    #include <clipping_planes_vertex>

    gl_Position = projectionMatrix * mvPosition;
    vColor = color;
}