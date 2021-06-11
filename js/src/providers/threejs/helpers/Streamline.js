// jshint maxstatements:false

const THREE = require('three');

module.exports = function (points, attributes, radius, radialSegments, color, verticesColors, colorRange) {
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const normals = [];
    const uvs = [];
    const colors = [];
    const indices = [];
    let lastTangent = null;
    let N = null;
    const vertex = new THREE.Vector3();
    let P1;
    let P2;
    let tangent;
    let mat;
    let vec2;
    let r;
    let theta;
    let offset;
    let originalP1;
    let originalP2;
    let start;
    let newRingCount = 0;

    function connectRings(from) {
        let a;
        let b;
        let c;
        let d;
        let k;

        for (let j = from; j < from + newRingCount - 1; j++) {
            for (k = 0; k < radialSegments; k++) {
                a = radialSegments * j + k;
                b = radialSegments * (j + 1) + k;
                c = radialSegments * (j + 1) + ((k + 1) % radialSegments);
                d = radialSegments * j + ((k + 1) % radialSegments);

                // faces
                indices.push(a, b, d);
                indices.push(b, c, d);
            }
        }
    }

    function makeRing(point1, point2, i, matrix) {
        const normal = new THREE.Vector3();
        const vec = new THREE.Vector3();
        const B = new THREE.Vector3();
        const ringTangent = point2.clone().sub(point1).normalize();
        let min;
        let tx;
        let ty;
        let tz;
        let v;
        let sin;
        let cos;

        if (N === null) {
            min = Number.MAX_VALUE;
            tx = Math.abs(ringTangent.x);
            ty = Math.abs(ringTangent.y);
            tz = Math.abs(ringTangent.z);

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

            vec.crossVectors(ringTangent, normal).normalize();

            N = new THREE.Vector3();
            N.crossVectors(ringTangent, vec);
        } else if (matrix) {
            N.applyMatrix4(matrix);
        }

        B.crossVectors(ringTangent, N);

        // generate normals and vertices for the current segment
        for (let j = 0; j < radialSegments; j++) {
            v = (j / radialSegments) * (Math.PI * 2);
            sin = Math.sin(v);
            cos = -Math.cos(v);

            // normal
            normal.x = (cos * N.x + sin * B.x);
            normal.y = (cos * N.y + sin * B.y);
            normal.z = (cos * N.z + sin * B.z);
            normal.normalize();
            normals.push(normal.x, normal.y, normal.z);

            // vertex
            vertex.x = point1.x + radius * normal.x;
            vertex.y = point1.y + radius * normal.y;
            vertex.z = point1.z + radius * normal.z;
            vertices.push(vertex.x, vertex.y, vertex.z);

            if (attributes !== null && attributes.length > 0) {
                uvs.push((attributes[i] - colorRange[0]) / (colorRange[1] - colorRange[0]), j / radialSegments);
            } else if (verticesColors !== null && verticesColors.length > 0) {
                colors.push(verticesColors[i * 3], verticesColors[i * 3 + 1], verticesColors[i * 3 + 2]);
            } else {
                colors.push(color.r, color.g, color.b);
            }
        }

        newRingCount++;
    }

    start = 0;

    for (let i = 0; i < points.length / 3; i++) {
        mat = null;

        if (i !== points.length / 3 - 1) {
            P1 = new THREE.Vector3().fromArray(points, i * 3);
            P2 = new THREE.Vector3().fromArray(points, i * 3 + 3);
            tangent = P2.clone().sub(P1).normalize();
        } else {
            P1 = new THREE.Vector3().fromArray(points, i * 3);
            P2 = new THREE.Vector3().fromArray(points, i * 3 - 3);
            P2.add(P1.clone().sub(P2).multiplyScalar(2.0));
        }

        if (Number.isNaN(P1.x) || Number.isNaN(P1.y) || Number.isNaN(P1.z)
            || Number.isNaN(P2.x) || Number.isNaN(P2.y) || Number.isNaN(P2.z)) {
            connectRings(start);
            lastTangent = null;
            N = null;
            start += newRingCount;
            newRingCount = 0;

            continue;
        }

        if (lastTangent && tangent) {
            vec2 = new THREE.Vector3().crossVectors(lastTangent, tangent);
            r = vec2.length();

            if (r > Number.EPSILON) {
                if (r > 0.3) {
                    offset = Math.min(radius, P2.clone().sub(P1).length() / 5.0);
                    originalP1 = P1.clone();
                    originalP2 = P2.clone();

                    P1.sub(lastTangent.clone().multiplyScalar(offset));
                    P2.copy(originalP1).add(lastTangent.clone().multiplyScalar(offset));
                    makeRing(P1, P2, i);

                    P1.copy(originalP1).add(tangent.clone().multiplyScalar(offset));
                    P2.copy(originalP2);
                }

                vec2 = vec2.divideScalar(r);

                theta = Math.acos(THREE.Math.clamp(lastTangent.clone().dot(tangent), -1, 1));
                mat = new THREE.Matrix4().makeRotationAxis(vec2, theta);
            }
        }

        makeRing(P1, P2, i, mat);
        lastTangent = tangent.clone();
    }

    connectRings(start);

    geometry.setIndex(indices);
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));

    if (attributes !== null && attributes.length > 0) {
        geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    } else {
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    }

    return geometry;
};
