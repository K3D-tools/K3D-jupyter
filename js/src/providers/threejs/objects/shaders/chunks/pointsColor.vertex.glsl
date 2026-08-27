kPerPointOpacity = 1.0;

#if K3D_PER_POINT_OPACITY == 1
kPerPointOpacity = opacities;
#endif

#if K3D_COLOR_MAP == 1
vec4 kTexel = texture2D(colormap, vec2((attributes - low) / (high - low), 0.5));

kPointColor = kTexel.rgb;
kPerPointOpacity *= kTexel.a;
#endif
