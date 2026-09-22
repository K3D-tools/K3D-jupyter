uniform sampler2D tDiffuse;
uniform vec2 uSize;
// 1 when the source holds straight colour and has to be weighted by its coverage first
uniform int uPremultiply;

// a premultiplied layer copied verbatim - compositing comes from blending
void main (void)
{
    vec4 color = texture2D(tDiffuse, gl_FragCoord.xy / uSize);

    if (color.a == 0.) discard;

    gl_FragColor = (uPremultiply == 1) ? vec4(color.rgb * color.a, color.a) : color;
}
