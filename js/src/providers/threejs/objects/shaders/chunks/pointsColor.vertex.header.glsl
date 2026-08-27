varying float kPerPointOpacity;

#if K3D_COLOR_MAP == 1
uniform sampler2D colormap;
uniform float low;
uniform float high;
attribute float attributes;
varying vec3 kPointColor;
#endif

#if K3D_PER_POINT_OPACITY == 1
attribute float opacities;
#endif
