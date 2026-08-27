{
    float kScaled = clamp((texture(volumeTexture, kLocalPosition).x - low) / (high - low), 0.01, 0.99);
    vec4 kTexel = texture2D(colormap, vec2(kScaled, 0.5));

    diffuseColor = vec4(kTexel.rgb, kTexel.a * opacity);
}
