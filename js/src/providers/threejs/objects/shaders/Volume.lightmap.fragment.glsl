#include <common>
#include <clipping_planes_pars_fragment>
#include <lights_pars_begin>

precision highp sampler3D;

varying vec2 vUv;

uniform vec3 lightMapSize;
uniform vec2 lightMapRenderTargetSize;
uniform vec3 volumeMapSize;
uniform sampler2D colormap;
uniform sampler3D volumeTexture;
uniform float low;
uniform float high;
uniform vec4 scale;
uniform vec4 translation;
uniform float samples;
uniform float alpha_coef;
uniform vec3 lightDirection;
uniform mat4 modelViewMatrix;

float inv_range;

struct Ray {
    vec3 origin;
    vec3 direction;
    vec3 inv_direction;
    int sign[3];
};

vec3 aabb[2] = vec3[2](
	vec3(-0.5, -0.5, -0.5),
	vec3(0.5, 0.5, 0.5)
);

Ray makeRay(vec3 origin, vec3 direction) {
    vec3 inv_direction = vec3(1.0) / direction;

    return Ray(
        origin,
        direction,
        inv_direction,
        int[3](
			((inv_direction.x < 0.0) ? 1 : 0),
			((inv_direction.y < 0.0) ? 1 : 0),
			((inv_direction.z < 0.0) ? 1 : 0)
		)
    );
}

/*
	From: https://github.com/hpicgs/cgsee/wiki/Ray-Box-Intersection-on-the-GPU
*/
void intersect(
    in Ray ray, in vec3 aabb[2],
    out float tmin, out float tmax
){
    float tymin, tymax, tzmin, tzmax;
    tmin = (aabb[ray.sign[0]].x - ray.origin.x) * ray.inv_direction.x;
    tmax = (aabb[1-ray.sign[0]].x - ray.origin.x) * ray.inv_direction.x;
    tymin = (aabb[ray.sign[1]].y - ray.origin.y) * ray.inv_direction.y;
    tymax = (aabb[1-ray.sign[1]].y - ray.origin.y) * ray.inv_direction.y;
    tzmin = (aabb[ray.sign[2]].z - ray.origin.z) * ray.inv_direction.z;
    tzmax = (aabb[1-ray.sign[2]].z - ray.origin.z) * ray.inv_direction.z;
    tmin = max(max(tmin, tymin), tzmin);
    tmax = min(min(tmax, tymax), tzmax);
}

void main() {
    vec2 sliceCount =  lightMapRenderTargetSize / lightMapSize.xy;
    float zidx = floor(vUv.x * lightMapRenderTargetSize.x / lightMapSize.x)  +
                 floor(vUv.y * lightMapRenderTargetSize.y / lightMapSize.y) * sliceCount.x;

    if (zidx > lightMapSize.z) {
        discard;
    }

    float x = mod(vUv.x * lightMapRenderTargetSize.x, lightMapSize.x) / lightMapSize.x;
    float y = mod(vUv.y * lightMapRenderTargetSize.y, lightMapSize.y) / lightMapSize.y;
    float z = zidx / lightMapSize.z;

    vec3 localPosition = vec3(x, y, z);

    // check if inside
    vec3 delta_step = vec3(1.0) / lightMapSize;
    vec3 s = step(delta_step, localPosition) - step(vec3(1.0) - delta_step, localPosition);

    if (s.x * s.y * s.z < 1.0) {
        discard;
    }

    // start intersection
	float tmin = 0.0;
	float tmax = 0.0;
    float reducedSamples = samples / 2.0;
    float dist;

    inv_range = 1.0 / (high - low);
    aabb[0] = aabb[0] * scale.xyz + translation.xyz;
    aabb[1] = aabb[1] * scale.xyz + translation.xyz;

	intersect(makeRay((localPosition - vec3(0.5)) * scale.xyz + translation.xyz, -lightDirection), aabb, tmin, tmax);

    float backoff = length(scale.xyz / volumeMapSize);

	if (tmin >= tmax) {
	    dist = 0.0;
	} else {
	    dist = abs(tmax);
	}

    dist += backoff;
	backoff *= 3.0;

    if (dist <= backoff) {
        discard;
    }
	vec3 textcoord_start = localPosition - dist * lightDirection / scale.xyz;
	vec3 textcoord_end = localPosition - backoff * lightDirection / scale.xyz;
	vec3 textcoord_delta = textcoord_end - textcoord_start;

    // @TODO: protection. Sometimes strange situation happen and sampleCount is out the limit
    int sampleCount = min(int(length(textcoord_delta) * reducedSamples), int(reducedSamples * 1.8));

    textcoord_delta = textcoord_delta / float(sampleCount);
    vec3 textcoord = textcoord_start - textcoord_delta;

    float textcoord_delta_step = length(textcoord_delta);
    float sum_density = 0.0;

    for(int count = 0; count < sampleCount; count++){
        textcoord += textcoord_delta;

        #if NUM_CLIPPING_PLANES > 0
            vec4 plane;
            vec3 pos = -vec3(modelViewMatrix * vec4(textcoord - vec3(0.5), 1.0));

            #pragma unroll_loop
            for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
                plane = clippingPlanes[ i ];
                if ( dot( pos, plane.xyz ) > plane.w ) continue;
            }
        #endif

        float px = texture(volumeTexture, textcoord).x;
        float scaled_px = (px - low) * inv_range;

        if(scaled_px > 0.0) {
            scaled_px = min(scaled_px, 0.99);
            float alpha = texture(colormap, vec2(scaled_px, 0.5)).a;

            float density = 1.0 - pow(1.0 - alpha, textcoord_delta_step * alpha_coef);
            density *= (1.0 - sum_density);
            sum_density += density;

            if(sum_density >= 0.95){
                sum_density = 1.0;
                break;
            }
        }
    }

    gl_FragColor = vec4(sum_density, 0, 0, 1.0);
}