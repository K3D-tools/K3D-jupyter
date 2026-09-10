// One 5x5 bilateral pass over the traced image, stopped by measured variance.
//
// The usual edge-stopping features - depth, normal, albedo - are meaningless inside a
// participating medium: there is no first surface, the first collision is a random variable, and
// at low sample counts every one of those buffers is itself noise. Two quantities do survive.
// Variance says whether a neighbour differs because the image differs there or because this pixel
// has not converged yet, and it is defined everywhere. Accumulated opacity says whether the
// neighbour is even the same matter - without it the filter cannot separate a dark pixel of the
// medium from the black background behind it, and pulls the medium towards zero.
//
// Neither is stored. Both are recomputed here from the two parity halves of the accumulation,
// which are already full-resolution textures: Var(mean) = (A - B)^2 nA nB / n^2.
//
// One pass, and a radius of two pixels, is deliberate. That is where Monte Carlo grain lives and
// where almost nothing else does. Widening it - an a-trous cascade reaches sixty pixels in five
// passes - lets colour cross half an object, and the image turns into a smooth grey lie long
// before the noise is gone.

uniform sampler2D tDiffuse;
uniform sampler2D tHalfA;
uniform sampler2D tHalfB;
uniform vec2 uSize;
// (nA * nB) / n^2, the factor that turns the half-difference into the variance of the mean
uniform float uVarWeight;
// how many standard deviations of difference the filter still treats as noise. Around 4 the bone
// in a CT scan starts to look waxy: the grain goes and the trabecular texture under it goes too.
uniform float uPhi;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

// The layer reaching this pass is premultiplied - rawBlit composites it with OneFactor over
// OneMinusSrcAlpha - and premultiplied colour is exactly the quantity that is linear under a
// weighted average. So the taps are summed as they are, with no conversion in either direction:
// un-premultiplying here divides thin regions by a small alpha and washes them out.
vec4 tap(sampler2D tex, ivec2 p) {
    return texelFetch(tex, p, 0);
}

// luminance variance in .x, opacity variance in .y
vec2 varianceAt(ivec2 p) {
    vec4 a = tap(tHalfA, p);
    vec4 b = tap(tHalfB, p);
    vec2 d = vec2(dot(a.rgb - b.rgb, LUMA), a.a - b.a);

    return d * d * uVarWeight;
}

void main() {
    ivec2 size = ivec2(uSize);
    ivec2 p = ivec2(gl_FragCoord.xy);

    // The variance estimate is itself a noisy quantity. Without this box the edge-stopping term
    // flickers from pixel to pixel and the filter leaves its own speckle behind.
    vec2 variance = vec2(0.0);

    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            variance += varianceAt(clamp(p + ivec2(i, j), ivec2(0), size - ivec2(1)));
        }
    }

    variance /= 9.0;

    vec4 centre = tap(tDiffuse, p);
    float centreLuma = dot(centre.rgb, LUMA);
    // the floor keeps a converged pixel from dividing by zero and blurring everything
    vec2 sigma = uPhi * sqrt(max(variance, vec2(0.0))) + vec2(1e-4, 1e-3);

    float kernel[5] = float[5](0.0625, 0.25, 0.375, 0.25, 0.0625);

    vec4 sum = vec4(0.0);
    float weightSum = 0.0;

    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < 5; i++) {
            ivec2 q = p + ivec2(i - 2, j - 2);

            if (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y) {
                continue;
            }

            vec4 neighbour = tap(tDiffuse, q);
            float weight = kernel[i] * kernel[j] * exp(
                -abs(dot(neighbour.rgb, LUMA) - centreLuma) / sigma.x
                - abs(neighbour.a - centre.a) / sigma.y
            );

            sum += weight * neighbour;
            weightSum += weight;
        }
    }

    gl_FragColor = sum / max(weightSum, 1e-8);
}
