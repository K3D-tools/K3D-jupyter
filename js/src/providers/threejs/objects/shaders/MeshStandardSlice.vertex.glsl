#include <common>
#include <color_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>

uniform vec4 slicePlanes[MAXIMUM_SLICE_PLANES];
uniform int slicePlanesCount;

attribute vec3 next1;
attribute vec3 next2;
varying float vectorVisible;

vec3 intersectPlane(vec3 p1, vec3 p2, vec4 p) {
    vec3 lineDirection = p2 - p1;

    float distance1 = abs(dot(p1, p.xyz) + p.w);
    float distance2 = abs(dot(p2, p.xyz) + p.w);

    return p1 + lineDirection * distance1 / (distance1 + distance2);
}

void main() {
    #include <color_vertex>

    vec3 transformed = vec3(position);
    vec3 transformedNext1 = vec3(next1);
    vec3 transformedNext2 = vec3(next2);

    vec4 mvPosition = vec4(transformed, 1.0);
    vec4 mvPositionNext1 = vec4(transformedNext1, 1.0);
    vec4 mvPositionNext2 = vec4(transformedNext2, 1.0);

    for (int i = 0; i < slicePlanesCount; i++) {

        vec4 slicePlane = slicePlanes[i];

        bool side1 = dot(-mvPosition.xyz, slicePlane.xyz) > slicePlane.w;
        bool side2 = dot(-mvPositionNext1.xyz, slicePlane.xyz) > slicePlane.w;
        bool side3 = dot(-mvPositionNext2.xyz, slicePlane.xyz) > slicePlane.w;

        if (!((side1 && side2 && side3) || (!side1 && !side2 && !side3))) {

            if (side1 ^^ side2) {
                mvPosition.xyz = intersectPlane(mvPosition.xyz, mvPositionNext1.xyz, slicePlane);
            }

            if (side1 ^^ side3) {
                mvPosition.xyz = intersectPlane(mvPosition.xyz, mvPositionNext2.xyz, slicePlane);
            }

            mvPosition = modelViewMatrix * mvPosition;

            vectorVisible = 1.0;

            gl_Position = projectionMatrix * mvPosition;

            #include <logdepthbuf_vertex>
            #include <worldpos_vertex>
            #include <clipping_planes_vertex>
            return;
        }
    }

    vectorVisible = 0.0;
    gl_Position = vec4(0.0);
}
