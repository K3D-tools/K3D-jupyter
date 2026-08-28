uniform vec2 uScreenSize;
uniform sampler2D uPrevDepthTexture;
uniform int uLayer;
uniform float uDepthOffset;

#if defined( K3D_PEEL_DEPTH_OUT )
// peel target attachment 1; dropped when the framebuffer has no draw buffer there
layout(location = 1) out highp vec4 k3dPeelDepth;
#endif
