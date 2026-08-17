#include <tonemapping_pars_fragment>

varying vec2 vUv;
uniform sampler2D uTextureA;
uniform sampler2D uTextureB;
uniform int uBlit;
uniform int uToneMapping;

// three bakes renderer.toneMapping into programs only when drawing to the canvas -
// none of our composition does, so the curve is applied here, on the final image
vec3 k3dToneMap(vec3 color) {
    if (uToneMapping == 1) {
        return AgXToneMapping(color);
    }
    if (uToneMapping == 2) {
        return ACESFilmicToneMapping(color);
    }

    return color;
}

void main(){
    vec4 src = texture2D(uTextureA, vUv);
    vec4 dst = texture2D(uTextureB, vUv);

    if (uBlit == 0) {
        gl_FragColor = src;

        // the accumulator is premultiplied - the curve expects straight colour
        if (uToneMapping != 0 && gl_FragColor.a > 0.) {
            gl_FragColor.xyz = k3dToneMap(gl_FragColor.xyz / gl_FragColor.a) * gl_FragColor.a;
        }
    } else if (uBlit == 2) {
        // volume segments arrive premultiplied by the ray march
        gl_FragColor = src;
    } else {
        gl_FragColor = src;
        gl_FragColor.xyz *= gl_FragColor.a;
    }

    if (gl_FragColor.a == 0.) discard;
}
