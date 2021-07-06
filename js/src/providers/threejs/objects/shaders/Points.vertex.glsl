uniform float size;
uniform float scale;

attribute vec3 color;

#if (USE_PER_POINT_OPACITY == 1)
attribute float opacities;
#endif

#if (USE_PER_POINT_SIZE == 1)
attribute float sizes;
#endif

#if (USE_COLOR_MAP == 1)
uniform sampler2D colormap;
uniform float low;
uniform float high;
attribute float attributes;
#endif

varying vec4 vColor;
varying vec4 mvPosition;

#include <clipping_planes_pars_vertex>

void main() {
    float perPointOpacity = 1.0;

    mvPosition = modelViewMatrix * vec4(position, 1.0);

    #if (USE_PER_POINT_SIZE == 1)
    gl_PointSize = 2.0 * sizes * (scale / - mvPosition.z);
    #else
    gl_PointSize = 2.0 * size * (scale / - mvPosition.z);
    #endif

    #include <clipping_planes_vertex>

    gl_Position = projectionMatrix * mvPosition;

    #if (USE_PER_POINT_OPACITY == 1)
    perPointOpacity = opacities;
    #endif

    #if (USE_COLOR_MAP == 1)
    float scaled_px = (attributes - low) / (high - low);
    vec4 finalSphereColor = texture2D(colormap, vec2(scaled_px, 0.5));

    finalSphereColor.a *= perPointOpacity;
    vColor = finalSphereColor;
    #else
    vColor = vec4(color, perPointOpacity);
    #endif
}