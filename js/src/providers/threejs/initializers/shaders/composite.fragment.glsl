varying vec2 vUv;
uniform sampler2D uTextureA;
uniform sampler2D uTextureB;
uniform int uBlit;

void main(){
    vec4 src = texture2D(uTextureA, vUv);
    vec4 dst = texture2D(uTextureB, vUv);

    if (uBlit == 0) {
        gl_FragColor = src;
    } else {
        gl_FragColor = src;
        gl_FragColor.xyz *= gl_FragColor.a;
    }

    if (gl_FragColor.a == 0.) discard;
}