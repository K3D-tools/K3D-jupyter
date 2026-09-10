precision highp sampler3D;

uniform sampler2D colormap;
uniform sampler3D volumeTexture;
uniform float low;
uniform float high;
varying vec3 kLocalPosition;

#include <k3d_color_range>
