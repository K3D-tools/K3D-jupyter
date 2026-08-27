#include <tonemapping_pars_fragment>

uniform sampler2D tDiffuse;
uniform vec2 uSize;
uniform int uToneMapping;

vec3 k3dToneMap(vec3 color) {
    if (uToneMapping == 1) {
        return AgXToneMapping(color);
    }
    if (uToneMapping == 2) {
        return ACESFilmicToneMapping(color);
    }

    return color;
}

// the intermediate target mirrors the destination 1:1, gl_FragCoord maps identically
void main (void)
{
    vec4 color = texture2D(tDiffuse, gl_FragCoord.xy / uSize);

    if (color.a == 0.) discard;

    gl_FragColor = vec4(k3dToneMap(color.rgb) * color.a, color.a);
}
