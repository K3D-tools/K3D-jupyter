//precision highp sampler3D;
//
//uniform sampler2D colormap;
//uniform sampler3D volumeTexture;
//uniform float low;
//uniform float high;
//uniform float opacity;
//
//varying vec3 localPosition;
//
//#include <common>
//#include <dithering_pars_fragment>
//#include <clipping_planes_pars_fragment>
//#include <logdepthbuf_pars_fragment>
//
//void main() {
//
//    #include <clipping_planes_fragment>
//    #include <logdepthbuf_fragment>
//
//    float inv_range = 1.0 / (high - low);
//    float px = texture(volumeTexture, localPosition).x;
//    float scaled_px = (px - low) * inv_range;
//
//    scaled_px = max(min(scaled_px, 0.99), 0.01);
//    vec4 texelColor = texture(colormap, vec2(scaled_px, 0.5));
//
//    texelColor.a *= opacity;
//
//    gl_FragColor = texelColor;
//
//    #include <premultiplied_alpha_fragment>
//    #include <dithering_fragment>
//}
//
//

uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;

precision highp sampler3D;

uniform sampler2D colormap;
uniform sampler3D volumeTexture;
uniform float low;
uniform float high;

varying vec3 localPosition;

#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <uv2_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
#include <normal_pars_fragment>

void main() {

    #include <clipping_planes_fragment>

    float inv_range = 1.0 / (high - low);
    float px = texture(volumeTexture, localPosition).x;
    float scaled_px = (px - low) * inv_range;
    scaled_px = max(min(scaled_px, 0.99), 0.01);
    vec4 diffuseColor = texture(colormap, vec2(scaled_px, 0.5));
    diffuseColor.a *= opacity;

    ReflectedLight reflectedLight = ReflectedLight(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
    vec3 totalEmissiveRadiance = emissive;

    #include <logdepthbuf_fragment>
    #include <map_fragment>
    #include <color_fragment>
    #include <alphamap_fragment>
    #include <alphatest_fragment>
    #include <specularmap_fragment>
    #include <normal_fragment_begin>
    #include <normal_fragment_maps>
    #include <emissivemap_fragment>

    // accumulation
    #include <lights_phong_fragment>
    #include <lights_fragment_begin>
    #include <lights_fragment_maps>
    #include <lights_fragment_end>

    // modulation
    #include <aomap_fragment>

    vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;

    #include <envmap_fragment>

    gl_FragColor = vec4(outgoingLight, diffuseColor.a);

    #include <tonemapping_fragment>
    #include <encodings_fragment>
    #include <fog_fragment>
    #include <premultiplied_alpha_fragment>
    #include <dithering_fragment>

}
