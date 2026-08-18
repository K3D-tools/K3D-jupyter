uniform sampler2D tDiffuse;
uniform vec2 uSize;

// verbatim copy of a premultiplied layer - compositing comes from blending
void main (void)
{
    vec4 color = texture2D(tDiffuse, gl_FragCoord.xy / uSize);

    if (color.a == 0.) discard;

    gl_FragColor = color;
}
