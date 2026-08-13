uniform sampler2D tAO;
uniform vec2 uUvScale;
uniform vec2 uUvBias;

void main (void)
{
    // gl_FragCoord mapped into the full-frame AO buffer: identity for the canvas,
    // strip offset/stretch for chunked screenshot targets
    float ao = texture2D(tAO, gl_FragCoord.xy * uUvScale + uUvBias).r;

    gl_FragColor = vec4(ao, ao, ao, 1.0);
}
