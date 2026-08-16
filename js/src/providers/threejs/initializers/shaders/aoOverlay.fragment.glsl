uniform sampler2D tAO;
uniform sampler2D tDepth;
uniform vec2 uUvScale;
uniform vec2 uUvBias;

void main (void)
{
    // gl_FragCoord mapped into the full-frame AO buffer: identity for the canvas,
    // strip offset/stretch for chunked screenshot targets
    vec2 uv = gl_FragCoord.xy * uUvScale + uUvBias;
    float ao = texture2D(tAO, uv).r;

    // volumetric shells (depth.g == 2.0): occlusion works at full strength down to a
    // floor. Without one, deep AO of a corrugated isosurface multiplies the whole ray,
    // haze included, and dark colormaps drop into black noise
    if (texture2D(tDepth, uv).g > 1.5) {
        ao = max(ao, 0.4);
    }

    gl_FragColor = vec4(ao, ao, ao, 1.0);
}
