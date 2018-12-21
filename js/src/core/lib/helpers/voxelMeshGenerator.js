//jshint maxstatements:false, maxcomplexity:false, maxdepth:false
//Performance reason

'use strict';


/**
 * Generate greedy voxel mesh
 * @method generateGreedyVoxelMesh
 * @memberof K3D.Helpers
 * @param {Array} voxels
 * @param {Array} colorMap
 * @param {Number} chunkSize
 * @param {Object} voxelSize
 * @param {Object} offsets
 * @param {Object} calculate_outlines
 * @return {Object} with two properties - vertices and colors
 */
function generateGreedyVoxelMesh(voxels, colorMap, chunkSize, voxelSize, offsets, calculate_outlines) {
    var vertices = [],
        colors = [],
        outlines = [],
        quadIndex = 0,
        outlineIndex = 0,
        width = voxelSize.width,
        height = voxelSize.height,
        length = voxelSize.length,
        dims = [width, height, length],
        mask = new Int32Array((chunkSize + 1) * (chunkSize + 1)),
        d,
        x, q,
        u, v,
        a, b,
        i, j, k,
        w, h, l,
        du, dv,
        n, c,
        idx, off,
        ending = [
            Math.min(offsets.x + chunkSize, width),
            Math.min(offsets.y + chunkSize, height),
            Math.min(offsets.z + chunkSize, length)
        ],
        mask_ending = [
            Math.min(offsets.x + chunkSize + 1, width),
            Math.min(offsets.y + chunkSize + 1, height),
            Math.min(offsets.z + chunkSize + 1, length)
        ];

    function prepareVertices(x, du, dv) {
        // Performance over readability
        return [
            [x[0] / width, x[1] / height, x[2] / length],
            [(x[0] + dv[0]) / width, (x[1] + dv[1]) / height, (x[2] + dv[2]) / length],
            [(x[0] + du[0] + dv[0]) / width, (x[1] + du[1] + dv[1]) / height, (x[2] + du[2] + dv[2]) / length],
            [(x[0] + du[0]) / width, (x[1] + du[1]) / height, (x[2] + du[2]) / length]
        ];
    }

    function computerHeight(w, c, j, n, u, v, ending, mask_ending, mask) {
        var h, k, off = n;

        for (h = 1; j + h < ending[v]; h++) {
            off += mask_ending[u] - offsets[u];
            for (k = 0; k < w; k++) {
                if (c !== mask[k + off]) {
                    return h;
                }
            }
        }

        return h;
    }

    function makeOutline(outlines, x, v, outlineIndex) {
        var index = outlineIndex * 6, offset = [0, 0, 0];
        offset[v] = 1;

        outlines[index] = x[0] / width;
        outlines[index + 1] = x[1] / height;
        outlines[index + 2] = x[2] / length;

        outlines[index + 3] = (x[0] + offset[0]) / width;
        outlines[index + 4] = (x[1] + offset[1]) / height;
        outlines[index + 5] = (x[2] + offset[2]) / length;
    }

    offsets = [offsets.x, offsets.y, offsets.z];

    //Sweep over 3-axes
    q = [1, width, width * height];

    for (d = 0; d < 3; d++) {
        x = [0, 0, 0];
        u = (d + 1) % 3;
        v = (d + 2) % 3;

        for (x[d] = -1 + offsets[d]; x[d] < ending[d];) {
            //Compute mask
            var maskFilled = false;

            for (x[v] = offsets[v], n = 0; x[v] < mask_ending[v]; x[v]++) {
                x[u] = offsets[u];

                idx = x[0] + width * (x[1] + height * x[2]);

                for (; x[u] < mask_ending[u]; x[u]++, n++, idx += q[u]) {
                    a = (0 <= x[d] ? voxels[idx] : -1);
                    b = (x[d] < dims[d] - 1 ? voxels[idx + q[d]] : -1);

                    if (a === b || (a > 0 && b > 0)) {
                        mask[n] = 0;
                    } else if (a > 0) {
                        mask[n] = a;
                        maskFilled = true;
                    } else {
                        mask[n] = b > 0 ? -b : 0;
                        maskFilled |= b > 0;
                    }
                }
            }

            x[d]++;

            if (!maskFilled) {
                continue;
            }

            // outlines
            if (calculate_outlines) {
                for (j = offsets[v], n = 0; j < ending[v]; j++) {
                    x[u] = offsets[u];
                    x[v] = j;

                    if (offsets[u] === 0 && mask[n] !== 0) {
                        makeOutline(outlines, x, v, outlineIndex++);
                    }

                    for (i = offsets[u]; i < mask_ending[u] - 1; i++, n++) {
                        x[u] = i + 1;
                        x[v] = j;

                        if (mask[n] !== mask[n + 1]) {
                            makeOutline(outlines, x, v, outlineIndex++);
                        }
                    }

                    x[u] = ending[u];

                    if (i === dims[u] - 1 && mask[n] !== 0) {
                        makeOutline(outlines, x, v, outlineIndex++);
                    }

                    n++;
                }
            }

            //Generate mesh for mask using lexicographic ordering
            for (j = offsets[v], n = 0; j < ending[v]; j++) {
                for (i = offsets[u]; i < ending[u];) {

                    c = mask[n];

                    if (c) {
                        //Compute width
                        w = 1;
                        while (c === mask[n + w] && i + w < ending[u]) {
                            w++;
                        }

                        //Compute height
                        h = computerHeight(w, c, j, n, u, v, ending, mask_ending, mask);

                        //Add quad
                        x[u] = i;
                        x[v] = j;

                        du = [0, 0, 0];
                        dv = [0, 0, 0];

                        if (c > 0) {
                            dv[v] = h;
                            du[u] = w;
                        } else {
                            c = -c;
                            du[v] = h;
                            dv[u] = w;
                        }

                        makeQuad(
                            prepareVertices(x, du, dv),
                            vertices,
                            colors,
                            prepareColor(colorMap, c),
                            quadIndex++
                        );

                        //Zero-out mask
                        off = n;
                        for (l = 0; l < h; l++) {
                            for (k = 0; k < w; k++) {
                                mask[off + k] = 0;
                            }
                            off += mask_ending[u] - offsets[u];
                        }

                        i += w;
                        n += w;
                    } else {
                        i++;
                        n++;
                    }
                }

                if (ending[u] != mask_ending[u]) {
                    n++;
                }
            }
        }
    }

    return {
        vertices: vertices,
        colors: colors,
        outlines: outlines
    };
}

/**
 * Generate culled voxel mesh
 * @method generateCulledVoxelMesh
 * @memberof K3D.Helpers
 * @param {Array} voxels
 * @param {Array} colorMap
 * @param {Number} chunkSize
 * @param {Object} voxelSize
 * @param {Object} offsets
 * @return {Object} with two properties - vertices and colors
 */
function generateCulledVoxelMesh(voxels, colorMap, chunkSize, voxelSize, offsets) {
    var x, y, z, i,
        vertices = [],
        colors = [],
        ending,
        width = voxelSize.width,
        height = voxelSize.height,
        length = voxelSize.length,
        quadIndex = 0;

    function prepareVertices(x, y, z) {
        // Performance over readability
        return [
            [(x + 1) / width, (y + 1) / height, (z + 1) / length],
            [x / width, (y + 1) / height, (z + 1) / length],
            [x / width, y / height, (z + 1) / length],
            [(x + 1) / width, y / height, (z + 1) / length],
            [(x + 1) / width, (y + 1) / height, z / length],
            [x / width, (y + 1) / height, z / length],
            [x / width, y / height, z / length],
            [(x + 1) / width, y / height, z / length]
        ];
    }

    function processBox(p, color, x, y, z, i) {
        /*
         * ASCII Art - vertices index in cube :D
         *
         *   5____4
         *  1/___0/|
         *  | 6__|_7
         *  2/___3/
         *
         */

        if (z === length - 1 || voxels[i + width * height] === 0) {
            makeQuad([p[1], p[0], p[3], p[2]], vertices, colors, color, quadIndex++);
        }

        if (x === width - 1 || voxels[i + 1] === 0) {
            makeQuad([p[0], p[4], p[7], p[3]], vertices, colors, color, quadIndex++);
        }

        if (z === 0 || voxels[i - width * height] === 0) {
            makeQuad([p[4], p[5], p[6], p[7]], vertices, colors, color, quadIndex++);
        }

        if (x === 0 || voxels[i - 1] === 0) {
            makeQuad([p[5], p[1], p[2], p[6]], vertices, colors, color, quadIndex++);
        }

        if (y === height - 1 || voxels[i + width] === 0) {
            makeQuad([p[5], p[4], p[0], p[1]], vertices, colors, color, quadIndex++);
        }

        if (y === 0 || voxels[i - width] === 0) {
            makeQuad([p[2], p[3], p[7], p[6]], vertices, colors, color, quadIndex++);
        }
    }

    ending = {
        x: Math.min(offsets.x + chunkSize, width),
        y: Math.min(offsets.y + chunkSize, height),
        z: Math.min(offsets.z + chunkSize, length)
    };

    for (z = offsets.z; z < ending.z; z++) {
        for (y = offsets.y; y < ending.y; y++) {
            for (x = offsets.x; x < ending.x; x++) {

                i = x + width * (y + height * z);

                if (voxels[i] !== 0) {
                    processBox(
                        prepareVertices(x, y, z),
                        prepareColor(colorMap, voxels[i]),
                        x, y, z, i
                    );
                }
            }
        }
    }

    return {
        vertices: vertices,
        colors: colors
    };
}

function prepareColor(colorMap, voxel) {
    var colorIndex = (voxel - 1) * 3;

    return [colorMap[colorIndex], colorMap[colorIndex + 1], colorMap[colorIndex + 2]];
}

function makeQuad(points, vertices, colors, color, quadIndex) {
    var index = quadIndex * 18, i;

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

    for (i = 0; i < 18; i += 3) {
        colors[index + i] = color[0];
        colors[index + i + 1] = color[1];
        colors[index + i + 2] = color[2];
    }
}


function initializeGreedyVoxelMesh(voxels, colorMap, chunkSize, voxelSize, offsets, calculate_outlines) {
    return generateGreedyVoxelMesh.bind(null, voxels, colorMap, chunkSize, voxelSize,
        offsets, calculate_outlines);
}

function initializeCulledVoxelMesh(voxels, colorMap, chunkSize, voxelSize, offsets) {
    return generateCulledVoxelMesh.bind(null, voxels, colorMap, chunkSize, voxelSize,
        offsets);
}

module.exports = {
    initializeGreedyVoxelMesh: initializeGreedyVoxelMesh,
    initializeCulledVoxelMesh: initializeCulledVoxelMesh,
    generateGreedyVoxelMesh: generateGreedyVoxelMesh,
    generateCulledVoxelMesh: generateCulledVoxelMesh
};

