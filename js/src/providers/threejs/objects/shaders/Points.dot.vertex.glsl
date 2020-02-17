uniform float size;
uniform float scale;

attribute vec3 color;
#if (USE_PER_POINT_OPACITY == 1)
attribute float opacities;
#endif

varying vec4 vColor;
varying vec4 mvPosition;

#include <clipping_planes_pars_vertex>

void main() {
    float perPointOpacity = 1.0;

    mvPosition = modelViewMatrix * vec4( position, 1.0 );
    gl_PointSize = 2.0 * size;

    #include <clipping_planes_vertex>

    gl_Position = projectionMatrix * mvPosition;

    #if (USE_PER_POINT_OPACITY == 1)
        perPointOpacity = opacities;
    #endif

    vColor = vec4(color, perPointOpacity);
}