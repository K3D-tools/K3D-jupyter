#include <common>

precision highp sampler3D;

uniform sampler2D colormap;
uniform sampler3D volumeTexture[TEXTURE_COUNT];
uniform sampler3D mask;

uniform float low[TEXTURE_COUNT];
uniform float high[TEXTURE_COUNT];
uniform vec3 volumeSize[TEXTURE_COUNT];
uniform float opacity;

uniform float maskOpacity;
uniform sampler2D maskColors;
uniform sampler2D activeMasks;
uniform int activeMasksCount;

varying vec3 coord;


#include <dithering_pars_fragment>
#include <clipping_planes_pars_fragment>
#include <logdepthbuf_pars_fragment>

vec4 cubicSample(sampler3D volume, vec3 u, vec3 R) {
    #if CUBIC > 0
    vec3 U = u * R + 0.5;
    vec3 F = fract(U);

    U = floor(U) + F * F * (3. - 2. * F);
    //    U = floor(U) + F*F*F*(F*(F*6.-15.)+10.);   // use if you want smooth gradients
    return texture(volume, (U - 0.5) / R);
    #else
    return texture(volume, u);
    #endif
}

void main() {

    #include <clipping_planes_fragment>
    #include <logdepthbuf_fragment>

    float px, scaled_px;
    vec2 cm_coord;

    #pragma unroll_loop_start
    for (int i = 0; i < TEXTURE_COUNT; i++) {
        px = cubicSample(volumeTexture[i], coord, volumeSize[i]).x;
        scaled_px = (px - low[i]) / (high[i] - low[i]);
        scaled_px = max(min(scaled_px, 0.99), 0.01);
        cm_coord[i] = scaled_px;
    }
    #pragma unroll_loop_end

    vec4 texelColor = texture(colormap, cm_coord);

    if (activeMasksCount > 0) {
        float maskValue = texture(mask, coord).r * 255.0;

        for (int i = 0;i < activeMasksCount; i++) {
            if (int(texture(activeMasks, vec2(float(i) / 255.0, 0.5)).b * 255.0) == int(maskValue)) {
                vec3 cm = texture(maskColors, vec2(maskValue / 255.0, 0.5)).rgb;

                texelColor.xyz = mix(texelColor.xyz, texelColor.xyz * cm, maskOpacity);
                break;
            }
        }
    }

    texelColor.a *= opacity;
    gl_FragColor = texelColor;

    #include <premultiplied_alpha_fragment>
    #include <dithering_fragment>
}
