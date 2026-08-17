#include <tonemapping_pars_fragment>

varying vec2 vUv;
uniform sampler2D uTextureA;
uniform sampler2D uTextureB;
uniform int uBlit;
uniform int uToneMapping;

// AO joins the composition per layer/segment, never the final blit: multiplying
// the finished image killed the volume light in front of dark meshes (a veil of
// gas over an occluded sphere vanished with the sphere's own AO)
uniform sampler2D tAO;
uniform sampler2D tAOVol;
uniform vec2 uAoScale;
uniform vec2 uAoBias;
uniform int uAoEnabled;

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

        if (uAoEnabled == 1) {
            // self-occlusion of the medium, floored - deep AO of a corrugated
            // isosurface would drop dark colormaps into black noise
            gl_FragColor.xyz *= max(texture2D(tAOVol, uAoBias + vUv * uAoScale).r, 0.4);
        }
    } else {
        gl_FragColor = src;
        gl_FragColor.xyz *= gl_FragColor.a;

        if (uAoEnabled == 1) {
            // geometry layers take the mesh-and-shell AO
            gl_FragColor.xyz *= texture2D(tAO, uAoBias + vUv * uAoScale).r;
        }
    }

    if (gl_FragColor.a == 0.) discard;
}
