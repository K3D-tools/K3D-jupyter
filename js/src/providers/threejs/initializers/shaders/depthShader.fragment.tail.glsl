if (uLayer != 0){
    vec2 screenPos = gl_FragCoord.xy * uScreenSize;

    float prevDepth = unpackRGBAToDepth(texture2D(uPrevDepthTexture, screenPos));


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
}
