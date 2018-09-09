#include <clipping_planes_pars_fragment>

precision highp sampler3D;

uniform mat4 transform;
uniform sampler3D volumeTexture;
uniform sampler2D colormap;
uniform sampler2D jitterTexture;
uniform float low;
uniform float high;
uniform mat4 modelViewMatrix;
uniform vec3 ambientLightColor;
uniform float samples;
uniform float alphaCorrection;

uniform vec4 scale;
uniform vec4 translation;

varying vec3 localPosition;
varying vec3 transformedCameraPosition;
varying vec3 transformedWorldPosition;

struct DirectionalLight {
    vec3 direction;
    vec3 color;

    int shadow;
    float shadowBias;
    float shadowRadius;
    vec2 shadowMapSize;
};

uniform DirectionalLight directionalLights[NUM_DIR_LIGHTS];

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
    float jitter = texture2D(jitterTexture, gl_FragCoord.xy/32.0).r;
	float tmin = 0.0;
	float tmax = 0.0;
    float px = 0.0;
    float inv_range = 1.0 / (high - low);
    vec4 pxColor = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 value = vec4(0.0, 0.0, 0.0, 0.0);

    aabb[0] = aabb[0] * scale.xyz + translation.xyz;
    aabb[1] = aabb[1] * scale.xyz + translation.xyz;

    vec3 direction = normalize(transformedWorldPosition - transformedCameraPosition);
	intersect(makeRay(transformedCameraPosition, direction), aabb, tmin, tmax);

    vec3 textcoord_end = localPosition + vec3(0.5);
	vec3 textcoord_start = textcoord_end - (tmax - max(0.0, tmin)) * direction.xyz / scale.xyz;
	vec3 textcoord_delta = textcoord_end - textcoord_start;

    // @TODO: protection. Sometimes strange situation happen and sampleCount is out the limit
    int sampleCount = min(int(length(textcoord_delta) * samples), int(samples * 1.8));
//    int sampleCount = int(length(textcoord_delta) * samples);

    textcoord_delta = textcoord_delta / float(sampleCount);
    textcoord_start = textcoord_start - textcoord_delta * (0.01 + 0.98 * jitter);

    vec3 textcoord = textcoord_start - textcoord_delta;

    float step = length(textcoord_delta);

	for(int count = 0; count < sampleCount; count++){
	    textcoord += textcoord_delta;

        #if NUM_CLIPPING_PLANES > 0
            vec4 plane;
            vec3 pos = -vec3(modelViewMatrix * vec4(textcoord, 1.0));

            #pragma unroll_loop
            for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
                plane = clippingPlanes[ i ];
                if ( dot( pos, plane.xyz ) > plane.w ) continue;
            }

            #if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
                bool clipped = true;

                #pragma unroll_loop
                for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
                    plane = clippingPlanes[ i ];
                    clipped = ( dot( pos, plane.xyz ) > plane.w ) && clipped;
                }

                if ( clipped ) continue;
            #endif
        #endif

		px = texture(volumeTexture, clamp(textcoord, 0.0, 0.99)).x;

		float scaled_px = (px - low) * inv_range;

		if(scaled_px > 0.0) {
		    scaled_px = min(scaled_px, 0.99);

            pxColor = texture(colormap, vec2(scaled_px, 0.5));
            pxColor.a = scaled_px;

            pxColor.a = 1.0 - pow(1.0 - pxColor.a, step * alphaCorrection);
            pxColor.a *= (1.0 - value.a);

            pxColor.rgb *= pxColor.a;

            // LIGHT
            #if NUM_DIR_LIGHTS > 0
                float gradientStep = 0.005;
                vec4 addedLights = vec4(ambientLightColor, 1.0);
                vec3 normal = normalize(vec3(
                    px -  texture(volumeTexture, textcoord + vec3(gradientStep,0,0)).x,
                    px -  texture(volumeTexture, textcoord + vec3(0,gradientStep,0)).x,
                    px -  texture(volumeTexture, textcoord + vec3(0,0,gradientStep)).x
                ));

                vec3 lightDirection;
                float lightingIntensity;

                #pragma unroll_loop
                for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
                    lightDirection = -directionalLights[ i ].direction;
                    lightingIntensity = clamp(dot(-lightDirection, normal), 0.0, 1.0);
                    addedLights.rgb += directionalLights[ i ].color * (0.05 + 0.95 * lightingIntensity);

                    #if (USE_SPECULAR == 1)
                    pxColor.rgb += directionalLights[ i ].color * pow(lightingIntensity, 10.0) * pxColor.a;
                    #endif
                }

                pxColor.rgb *= addedLights.xyz;
            #endif

            value += pxColor;

            if(value.a >= 0.95){
                value.a = 1.0;
                break;
            }
		}
	}

    gl_FragColor = value;
}