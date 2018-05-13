'use strict';
//jshint maxstatements:false

module.exports = function (points, attributes, radius, radialSegments, color, verticesColors, colorRange) {

    var geometry = new THREE.BufferGeometry(),
        i, j,
        vertices = [],
        normals = [],
        uvs = [],
        colors = [],
        indices = [],
        last_tangent = null,
        N = null,
        vertex = new THREE.Vector3();

    function makeSegment(P1, P2, i, mat) {
        var normal = new THREE.Vector3();
        var vec = new THREE.Vector3();
        var B = new THREE.Vector3();
        var tangent = P2.clone().sub(P1).normalize();

        if (N === null) {
            var min = Number.MAX_VALUE;
            var tx = Math.abs(tangent.x);
            var ty = Math.abs(tangent.y);
            var tz = Math.abs(tangent.z);

            if (tx <= min) {
                min = tx;
                normal.set(1, 0, 0);
            }

            if (ty <= min) {
                min = ty;
                normal.set(0, 1, 0);
            }

            if (tz <= min) {
                normal.set(0, 0, 1);
            }

            vec.crossVectors(tangent, normal).normalize();

            N = new THREE.Vector3();
            N.crossVectors(tangent, vec);
        } else if (mat) {
            N.applyMatrix4(mat);
        }

        B.crossVectors(tangent, N);

        // generate normals and vertices for the current segment
        for (j = 0; j < radialSegments; j++) {
            var v = j / radialSegments * Math.PI * 2;
            var sin = Math.sin(v);
            var cos = -Math.cos(v);

            // normal
            normal.x = ( cos * N.x + sin * B.x );
            normal.y = ( cos * N.y + sin * B.y );
            normal.z = ( cos * N.z + sin * B.z );
            normal.normalize();
            normals.push(normal.x, normal.y, normal.z);

            // vertex
            vertex.x = P1.x + radius * normal.x;
            vertex.y = P1.y + radius * normal.y;
            vertex.z = P1.z + radius * normal.z;
            vertices.push(vertex.x, vertex.y, vertex.z);

            if (attributes !== null && attributes.length > 0) {
                uvs.push((attributes[i] - colorRange[0]) / (colorRange[1] - colorRange[0]), j / radialSegments);
            } else if (verticesColors !== null && verticesColors.length > 0) {
                colors.push(verticesColors[i * 3], verticesColors[i * 3 + 1], verticesColors[i * 3 + 2]);
            } else {
                colors.push(color.r, color.g, color.b);
            }
        }
    }

    for (i = 0; i < points.length / 3; i++) {
        var P1, P2, tangent, mat = null;

        if (i !== points.length / 3 - 1) {
            P1 = new THREE.Vector3().fromArray(points, i * 3);
            P2 = new THREE.Vector3().fromArray(points, i * 3 + 3);
            tangent = P2.clone().sub(P1).normalize();
        } else {
            P1 = new THREE.Vector3().fromArray(points, i * 3);
            P2 = new THREE.Vector3().fromArray(points, i * 3 - 3);
            P2.add(P1.clone().sub(P2).multiplyScalar(2.0));
        }

        if (last_tangent && tangent) {
            var vec2 = new THREE.Vector3().crossVectors(last_tangent, tangent);
            var r = vec2.length();

            if (r > Number.EPSILON) {
                if (r > 0.3) {
                    var offset = Math.min(radius, P2.clone().sub(P1).length() / 5.0);
                    var originalP1 = P1.clone();
                    var originalP2 = P2.clone();

                    P1.sub(last_tangent.clone().multiplyScalar(offset));
                    P2.copy(originalP1).add(last_tangent.clone().multiplyScalar(offset));
                    makeSegment(P1, P2, i);

                    P1.copy(originalP1).add(tangent.clone().multiplyScalar(offset));
                    P2.copy(originalP2);
                }

                vec2 = vec2.divideScalar(r);

                var theta = Math.acos(THREE.Math.clamp(last_tangent.clone().dot(tangent), -1, 1));
                mat = new THREE.Matrix4().makeRotationAxis(vec2, theta);
            }
        }

        makeSegment(P1, P2, i, mat);

        last_tangent = tangent.clone();
    }

    for (j = 0; j < (vertices.length / 3 / radialSegments) - 1; j++) {
        for (i = 0; i < radialSegments; i++) {
            var a = radialSegments * j + i;
            var b = radialSegments * (j + 1) + i;
            var c = radialSegments * (j + 1) + ((i + 1) % radialSegments);
            var d = radialSegments * j + ((i + 1) % radialSegments);

            // faces
            indices.push(a, b, d);
            indices.push(b, c, d);
        }
    }

    geometry.setIndex(indices);
    geometry.addAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.addAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));

    if (attributes !== null && attributes.length > 0) {
        geometry.addAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    } else {
        geometry.addAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    }

    return geometry;
};
