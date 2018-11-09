#include <common>

varying vec2 vUv;

void main() {
//	#include <begin_vertex>
//	#include <project_vertex>

    gl_Position = projectionMatrix * vec4( position, 1.0 );
    vUv = uv;
}