#include <clipping_planes_pars_fragment>

precision highp sampler3D;

uniform mat4 transform;
uniform sampler3D volumeTexture;
uniform sampler2D colormap;
uniform float low;
uniform float high;
uniform mat4 modelViewMatrix;
uniform vec3 ambientLightColor;
uniform float samples_per_unit;

varying vec4 worldPosition;
varying vec3 localPosition;

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
    vec3 direction = normalize(worldPosition.xyz-cameraPosition);
	float tmin = 0.0;
	float tmax = 0.0;

	intersect(makeRay(cameraPosition, direction), aabb, tmin, tmax);

    vec3 textcoord = localPosition + vec3(0.5);

	vec3 start = textcoord - (tmax - tmin) * direction.xyz;
	vec3 end = textcoord;

    float len = distance(end, start);
    int sampleCount = int(len * samples_per_unit);
    vec3 texCo = vec3(0.0, 0.0, 0.0);
    vec4 pxColor = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 value = vec4(0.0, 0.0, 0.0, 0.0);
    float px = 0.0;

	for(int count = 0; count < sampleCount; count++){
        #if NUM_CLIPPING_PLANES > 0
            vec4 plane;
            vec3 pos = mix(cameraPosition + tmin * direction.xyz, cameraPosition + tmax * direction.xyz,
                           float(count)/float(sampleCount));

            pos = -vec3(modelViewMatrix * vec4(pos, 1.0));

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

		texCo = mix(start, end, float(count)/float(sampleCount));
		px = texture(volumeTexture, texCo).x;

		float scaled_px = (px - low) / (high - low);

		if(scaled_px > 0.0 && scaled_px < 1.0) {
            pxColor = texture(colormap, vec2(scaled_px, 0.5));

            float alpha = (scaled_px - 0.5) * 2.0;
            pxColor.a = clamp(alpha, 0.0, 1.0);

            pxColor.rgb = pxColor.rgb * pxColor.rgb * pxColor.a;

            // LIGHT
            float gradientStep = 0.005;
            vec4 addedLights = vec4(ambientLightColor, 1.0);
            vec3 normal = normalize(vec3(
                px -  texture(volumeTexture, texCo + vec3(gradientStep,0,0)).x,
                px -  texture(volumeTexture, texCo + vec3(0,gradientStep,0)).x,
                px -  texture(volumeTexture, texCo + vec3(0,0,gradientStep)).x
            ));

            for(int l = 0; l <NUM_DIR_LIGHTS; l++) {
                vec3 lightDirection = -directionalLights[l].direction;
                float lightingIntensity = clamp(dot(-lightDirection, normal), 0.0, 1.0);
                addedLights.rgb += directionalLights[l].color * (0.05 + 0.95 * lightingIntensity);

                #if (USE_SPECULAR == 1)
                pxColor.rgb += directionalLights[l].color * pow(lightingIntensity, 10.0) * pxColor.a;
                #endif
            }

            pxColor.rgb *= addedLights.xyz;

            value = value + pxColor - pxColor * value.a;

            if(value.a >= 0.95){
                value.a = 1.0;
                break;
            }
		}
	}

    gl_FragColor = value;
}