varying vec2 vUv;

void main(){
    vUv = uv;
    gl_Position = vec4(position.xy,0.,1.);
}