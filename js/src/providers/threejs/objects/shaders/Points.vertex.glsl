uniform float size;
uniform float scale;

attribute vec3 color;

varying vec3 vColor;
varying vec4 mvPosition;

void main() {
    mvPosition = modelViewMatrix * vec4( position, 1.0 );
    gl_PointSize = size * ( scale / - mvPosition.z );
    gl_Position = projectionMatrix * mvPosition;
    vColor = color;
}