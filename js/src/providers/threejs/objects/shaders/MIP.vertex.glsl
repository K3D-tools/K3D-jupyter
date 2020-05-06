uniform vec4 rotation;
uniform vec4 translation;

varying vec4 worldPosition;
varying vec3 localPosition;
varying vec3 transformedCameraPosition;
varying vec3 transformedWorldPosition;

vec3 rotate_vertex_position(vec3 pos, vec3 t, vec4 q) {
    vec3 p = pos.xyz - t.xyz;

    return p.xyz + 2.0 * cross(cross(p.xyz, q.xyz) + q.w * p.xyz, q.xyz) + t.xyz;
}

void main() {
    vec3 p;

	#include <begin_vertex>
	#include <project_vertex>

    localPosition = position;
    worldPosition = modelMatrix * vec4(transformed, 1.0);

    transformedCameraPosition = rotate_vertex_position(cameraPosition.xyz, translation.xyz, rotation);
    transformedWorldPosition = rotate_vertex_position(worldPosition.xyz, translation.xyz, rotation);

	#include <worldpos_vertex>
}