if (uLayer != 0){
    vec2 screenPos = gl_FragCoord.xy * uScreenSize;

    float prevDepth = unpackRGBAToDepth(texture2D(uPrevDepthTexture, screenPos));

    if (prevDepth + uDepthOffset - gl_FragCoord.z >= 0.){
        discard;
    }
}
}
