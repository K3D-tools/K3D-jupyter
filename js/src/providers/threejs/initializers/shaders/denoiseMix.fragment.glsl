// cinematic_denoise below 1: the traced and the denoised image, mixed in linear premultiplied space
uniform sampler2D tRaw;
uniform sampler2D tDenoised;
uniform float uMix;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);

    gl_FragColor = mix(texelFetch(tRaw, p, 0), texelFetch(tDenoised, p, 0), uMix);
}
