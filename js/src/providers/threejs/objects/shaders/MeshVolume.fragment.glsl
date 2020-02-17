precision highp sampler3D;

uniform sampler2D colormap;
uniform sampler3D volumeTexture;
uniform float low;
uniform float high;
uniform vec3 b1;
uniform vec3 b2;
uniform float opacity;

varying vec4 worldPosition;

#include <common>
#include <dithering_pars_fragment>
#include <clipping_planes_pars_fragment>

void main() {

	#include <clipping_planes_fragment>

    vec3 coord = (worldPosition.xyz - b1) / (b2 - b1);
    float inv_range = 1.0 / (high - low);
    float px = texture(volumeTexture, coord).x;
    float scaled_px = (px - low) * inv_range;

    scaled_px = max(min(scaled_px, 0.99), 0.01);
    vec4 texelColor = texture(colormap, vec2(scaled_px, 0.5));

    texelColor.a *= opacity;

	gl_FragColor = texelColor;

	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}
