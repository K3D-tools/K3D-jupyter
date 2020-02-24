varying vec2 vUv;
uniform sampler2D uTextureA;
uniform sampler2D uTextureB;
uniform int uBlit;

void main(){
    vec4 src = texture2D(uTextureA,vUv);
    vec4 dst = texture2D(uTextureB,vUv);

    float a1 = 1.-src.a;
    gl_FragColor.a = src.a + a1 * dst.a;
    gl_FragColor.rgb = src.rgb + a1 * dst.rgb;
    gl_FragColor.rgb /= gl_FragColor.a;
}