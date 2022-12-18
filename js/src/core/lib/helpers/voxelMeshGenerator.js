/* eslint no-shadow: 0 */

function prepareColor(colorMap, voxel) {
    if (Array.isArray(voxel)) {
        const ci1 = (Math.abs(voxel[0]) - 1) * 3;
        const ci2 = (Math.abs(voxel[1]) - 1) * 3;

        return [
            (colorMap[ci1] + colorMap[ci2]) / 2,
            (colorMap[ci1 + 1] + colorMap[ci2 + 1]) / 2,
            (colorMap[ci1 + 2] + colorMap[ci2 + 2]) / 2,
        ];
    }
    const colorIndex = (Math.abs(voxel) - 1) * 3;
    return [colorMap[colorIndex], colorMap[colorIndex + 1], colorMap[colorIndex + 2]];
}

function makeQuad(points, vertices, colors, color, quadIndex) {
    const index = quadIndex * 18;

    vertices[index] = points[0][0];
    vertices[index + 1] = points[0][1];
    vertices[index + 2] = points[0][2];
    vertices[index + 3] = points[3][0];
    vertices[index + 4] = points[3][1];
    vertices[index + 5] = points[3][2];
    vertices[index + 6] = points[1][0];
    vertices[index + 7] = points[1][1];
    vertices[index + 8] = points[1][2];

    vertices[index + 9] = points[1][0];
    vertices[index + 10] = points[1][1];
    vertices[index + 11] = points[1][2];
    vertices[index + 12] = points[3][0];
    vertices[index + 13] = points[3][1];
    vertices[index + 14] = points[3][2];
    vertices[index + 15] = points[2][0];
    vertices[index + 16] = points[2][1];
    vertices[index + 17] = points[2][2];

    for (let i = 0; i < 18; i += 3) {
        colors[index + i] = color[0];
        colors[index + i + 1] = color[1];
        colors[index + i + 2] = color[2];
    }
}

/**
 * Generate greedy voxel mesh
 * @method generateGreedyVoxelMesh
 * @memberof K3D.Helpers
 * @param {Array} chunk
 * @param {Array} colorMap
 * @param {Object} voxelSize
 * @param {Object} calculateOutlines
 * @param {Bool} transparent
 * @return {Object} with two properties - vertices and colors
 */
function generateGreedyVoxelMesh(chunk, colorMap, voxelSize, calculateOutlines, transparent) {
    const vertices = [];
    const colors = [];
    const outlines = [];
    let quadIndex = 0;
    let outlineIndex = 0;
    const {width} = voxelSize;
    const {height} = voxelSize;
    const {length} = voxelSize;
    const dims = [width, height, length];
    const voxelsIsArray = chunk.voxels instanceof Uint8Array;
    const maxSize = Math.max.apply(null, chunk.size);
    const mask = new Array((maxSize + 1) * (maxSize + 1)).fill(0);
    let d;
    let x;
    let u;
    let v;
    let a;
    let b;
    let i;
    let j;
    let k;
    let w;
    let h;
    let du;
    let dv;
    let n;
    let c;
    let idx;
    let off;
    const ending = [
        chunk.offset[0] + chunk.size[0],
        chunk.offset[1] + chunk.size[1],
        chunk.offset[2] + chunk.size[2],
    ];
    const extendedEnding = [
        Math.min(ending[0] + 1, width),
        Math.min(ending[1] + 1, height),
        Math.min(ending[2] + 1, length),
    ];

    function prepareVertices(x, du, dv) {
        // Performance over readability
        return [
            [
                (x[0] - chunk.offset[0]) / width,
                (x[1] - chunk.offset[1]) / height,
                (x[2] - chunk.offset[2]) / length,
            ],
            [
                (x[0] - chunk.offset[0] + dv[0]) / width,
                (x[1] - chunk.offset[1] + dv[1]) / height,
                (x[2] - chunk.offset[2] + dv[2]) / length,
            ],
            [
                (x[0] - chunk.offset[0] + du[0] + dv[0]) / width,
                (x[1] - chunk.offset[1] + du[1] + dv[1]) / height,
                (x[2] - chunk.offset[2] + du[2] + dv[2]) / length,
            ],
            [
                (x[0] - chunk.offset[0] + du[0]) / width,
                (x[1] - chunk.offset[1] + du[1]) / height,
                (x[2] - chunk.offset[2] + du[2]) / length],
        ];
    }

    function computerHeight(w, c, j, n, u, v, ending, maskEnding, mask) {
        let h;
        let k;
        let
            off = n;

        for (h = 1; j + h < ending[v]; h++) {
            off += maskEnding[u] - chunk.offset[u];
            for (k = 0; k < w; k++) {
                if (c !== mask[k + off]) {
                    return h;
                }
            }
        }

        return h;
    }

    function makeOutline(outlines, x, v, outlineIndex) {
        const index = outlineIndex * 6;
        const
            offset = [0, 0, 0];
        offset[v] = 1;

        outlines[index] = (x[0] - chunk.offset[0]) / width;
        outlines[index + 1] = (x[1] - chunk.offset[1]) / height;
        outlines[index + 2] = (x[2] - chunk.offset[2]) / length;

        outlines[index + 3] = (x[0] - chunk.offset[0] + offset[0]) / width;
        outlines[index + 4] = (x[1] - chunk.offset[1] + offset[1]) / height;
        outlines[index + 5] = (x[2] - chunk.offset[2] + offset[2]) / length;
    }

    // Sweep over 3-axes
    const q = [1, width, width * height];
    const qxyz = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

    for (d = 0; d < 3; d++) {
        x = [0, 0, 0];
        u = (d + 1) % 3;
        v = (d + 2) % 3;

        for (x[d] = -1 + chunk.offset[d]; x[d] < ending[d];) {
            // Compute mask
            let maskFilled = false;

            // compute last layer only if we are on edge of whole data
            for (x[v] = chunk.offset[v], n = 0; x[v] < extendedEnding[v]; x[v]++) {
                x[u] = chunk.offset[u];

                idx = x[0] + width * (x[1] + height * x[2]);

                for (; x[u] < extendedEnding[u]; x[u]++, n++, idx += q[u]) {
                    if (voxelsIsArray) {
                        a = (x[d] >= 0 ? chunk.voxels[idx] : -1);
                        b = (x[d] < dims[d] - 1 ? chunk.voxels[idx + q[d]] : -1);
                    } else {
                        a = (x[d] >= 0 ? chunk.voxels.get(x[0], x[1], x[2]) : -1);
                        b = (x[d] < dims[d] - 1
                            ? chunk.voxels.get(x[0] + qxyz[d][0], x[1] + qxyz[d][1], x[2] + qxyz[d][2])
                            : -1);
                    }

                    if (a === b || (a > 0 && b > 0 && !transparent)) {
                        mask[n] = 0;
                    } else if (a > 0) {
                        mask[n] = transparent && b > 0 ? [a, b] : a;
                        maskFilled = true;
                    } else {
                        mask[n] = b > 0 ? -b : 0;
                        maskFilled |= b > 0;
                    }
                }
            }

            x[d] += 1;

            if (!maskFilled) {
                continue;
            }

            // var str = "";
            // for (var px = chunk.offset[v], po = 0; px < extendedEnding[v]; px++) {
            //     var line = "";
            //     for (var py = chunk.offset[u]; py < extendedEnding[u]; py++, po++) {
            //         line = line + (mask[po] === -1 ? "#" : mask[po].toString(10));
            //     }
            //     str = "\n" + line + str;
            // }
            // console.log(str);

            // outlines
            if (calculateOutlines) {
                for (j = chunk.offset[v], n = 0; j < ending[v]; j++) {
                    x[u] = chunk.offset[u];
                    x[v] = j;

                    if (chunk.offset[u] === 0 && mask[n] !== 0) {
                        makeOutline(outlines, x, v, outlineIndex++);
                    }

                    for (i = chunk.offset[u]; i < extendedEnding[u] - 1; i++, n++) {
                        x[u] = i + 1;

                        if (mask[n] !== mask[n + 1]) {
                            makeOutline(outlines, x, v, outlineIndex++);
                        }
                    }

                    x[u] = ending[u];

                    if (i === dims[u] - 1 && mask[n] !== 0) {
                        makeOutline(outlines, x, v, outlineIndex++);
                    }

                    n += 1;
                }
            }

            // Generate mesh for mask using lexicographic ordering
            for (j = chunk.offset[v], n = 0; j < ending[v]; j++) {
                for (i = chunk.offset[u]; i < ending[u];) {
                    c = mask[n];

                    if (c) {
                        // Compute width
                        w = 1;
                        while (c === mask[n + w] && i + w < ending[u]) {
                            w += 1;
                        }

                        // Compute height
                        h = computerHeight(w, c, j, n, u, v, ending, extendedEnding, mask);

                        // Add quad
                        x[u] = i;
                        x[v] = j;

                        du = [0, 0, 0];
                        dv = [0, 0, 0];

                        if (c > 0) {
                            dv[v] = h;
                            du[u] = w;
                        } else {
                            // c = -c;
                            du[v] = h;
                            dv[u] = w;
                        }

                        makeQuad(
                            prepareVertices(x, du, dv),
                            vertices,
                            colors,
                            prepareColor(colorMap, mask[n]),
                            quadIndex++,
                        );

                        // Zero-out mask
                        off = n;
                        for (let l = 0; l < h; l++) {
                            for (k = 0; k < w; k++) {
                                mask[off + k] = 0;
                            }
                            off += extendedEnding[u] - chunk.offset[u];
                        }

                        i += w;
                        n += w;
                    } else {
                        i += 1;
                        n += 1;
                    }
                }

                if (ending[u] !== extendedEnding[u]) {
                    n += 1;
                }
            }
        }
    }

    return {
        offset: chunk.offset,
        vertices,
        colors,
        outlines,
    };
}

function initializeGreedyVoxelMesh(chunk, colorMap, voxelSize, calculateOutlines, transparent) {
    return generateGreedyVoxelMesh.bind(null, chunk, colorMap, voxelSize, calculateOutlines, transparent);
}

module.exports = {
    initializeGreedyVoxelMesh,
    generateGreedyVoxelMesh,
};
