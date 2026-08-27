uniform sampler2D tAO;
uniform sampler2D tAOVol;
uniform sampler2D tDepth;
uniform vec2 uUvScale;
uniform vec2 uUvBias;

void main (void)
{
    // gl_FragCoord mapped into the full-frame AO buffer: identity for the canvas,
    // strip offset/stretch for chunked screenshot targets
    vec2 uv = gl_FragCoord.xy * uUvScale + uUvBias;
    float ao = texture2D(tAO, uv).r;

    // volumetric shells (depth.g == 2.0) take AO computed from the shells alone -
    // meshes must not cast onto the whole ray integral. The floor keeps deep AO of a
    // corrugated isosurface from dropping dark colormaps into black noise
    if (texture2D(tDepth, uv).g > 1.5) {
        ao = max(texture2D(tAOVol, uv).r, 0.4);
    }

    gl_FragColor = vec4(ao, ao, ao, 1.0);
}
