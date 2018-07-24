varying vec4 worldPosition;
varying vec3 localPosition;

void main() {
	#include <begin_vertex>
	#include <project_vertex>

    localPosition = position;
    worldPosition = modelMatrix * vec4(transformed, 1.0);

	#include <worldpos_vertex>
}