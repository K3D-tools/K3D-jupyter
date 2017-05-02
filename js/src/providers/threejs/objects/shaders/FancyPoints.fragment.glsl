precision mediump float;
precision mediump int;

varying lowp vec3 vColor;

void main (void)
{
    vec2 uv = gl_PointCoord.xy;

    uv -= vec2(0.5, 0.5);

    float dist = sqrt(dot(uv, uv));
    float t = dist * 2.0;

    if (t > 1.0) discard;

    gl_FragColor = vec4(vColor, 1.0);
}
