precision mediump float;
precision mediump int;

uniform mat4 modelViewMatrix; // optional
uniform mat4 projectionMatrix; // optional

uniform float size;
uniform float scale;

attribute vec3 position;
attribute vec3 color;

varying lowp vec3 vColor;

void main() {
    vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
    gl_PointSize = size * ( scale / - mvPosition.z );

    gl_Position = projectionMatrix * mvPosition;
    vColor = color;
}