varying vec3 vColor;

void main (void)
{
    vec2 impostorSpaceCoordinate = (gl_PointCoord.xy - vec2(0.5, 0.5));
    float distanceFromCenter = length(impostorSpaceCoordinate);

    if (distanceFromCenter > 0.5) discard;

    gl_FragColor = vec4(vColor, 1.0);
}
