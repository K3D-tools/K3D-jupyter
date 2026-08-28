if (uLayer != 0){
    vec2 screenPos = gl_FragCoord.xy * uScreenSize;

    float prevDepth = texture2D(uPrevDepthTexture, screenPos).r;


    #if (PROVIDED_FRAG_COORD_Z > 0)
    if (prevDepth + uDepthOffset - fragCoordZ  >= 0.){
        discard;
    }
    #else
    if (prevDepth + uDepthOffset - gl_FragCoord.z >= 0.){
        discard;
    }
    #endif
}

#if defined( K3D_PEEL_DEPTH_OUT )
k3dPeelDepth = vec4( gl_FragCoord.z, 0.0, 0.0, 1.0 );
#endif
}
