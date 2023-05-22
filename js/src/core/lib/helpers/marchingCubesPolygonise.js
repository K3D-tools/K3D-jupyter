// jshint maxstatements:false, maxcomplexity:false, bitwise:false
// Performance reason

/**
 * Polygonise single marching cube
 * @method marchingCubesPolygonise
 * @memberof K3D.Helpers
 * @param {Array} positions
 * @param {Array} field scalar field R^3
 * @param {Number} width number of elements in edge of box
 * @param {Number} height number of elements in edge of box
 * @param {Number} length number of elements in edge of box
 * @param {Number} iso isolation threshold
 * @param {Number} x x coordinate of current cube
 * @param {Number} y y coordinate of current cube
 * @param {Number} z z coordinate of current cube
 * @return {Array} positions of vertices
 */

const vertexList = new Float32Array(12 * 3);
const fieldNeighbours = new Float32Array(8);
const {edgeTable} = require('./marchingCubesLookupTables');
const {triangleTable} = require('./marchingCubesLookupTables');

module.exports = function (positions, field, iso,
                           width, height, length,
                           k, j, i,
                           x, y, z,
                           sx, sy, sz) {
    let index = 0;
    const ptr = i * width * height + j * width + k;
    let o = 0;

    fieldNeighbours[0] = field[ptr];
    fieldNeighbours[1] = field[ptr + 1];
    fieldNeighbours[2] = field[ptr + width * height + 1];
    fieldNeighbours[3] = field[ptr + width * height];
    fieldNeighbours[4] = field[ptr + width];
    fieldNeighbours[5] = field[ptr + width + 1];
    fieldNeighbours[6] = field[ptr + width + width * height + 1];
    fieldNeighbours[7] = field[ptr + width + width * height];

    for (k = 0; k < 8; k++) {
        if (fieldNeighbours[k] < iso) {
            index |= 1 << k;
        }
    }

    const bits = edgeTable[index];

    if (bits === 0) {
        return;
    }

    if (bits & 1) {
        vertexList[0] = x + (sx * (iso - fieldNeighbours[0])) / (fieldNeighbours[1] - fieldNeighbours[0]);
        vertexList[1] = y;
        vertexList[2] = z;
    }

    if (bits & 2) {
        vertexList[3] = x + sx;
        vertexList[4] = y;
        vertexList[5] = z + (sz * (iso - fieldNeighbours[1])) / (fieldNeighbours[2] - fieldNeighbours[1]);
    }

    if (bits & 4) {
        vertexList[6] = x + (sx * (iso - fieldNeighbours[3])) / (fieldNeighbours[2] - fieldNeighbours[3]);
        vertexList[7] = y;
        vertexList[8] = z + sz;
    }

    if (bits & 8) {
        vertexList[9] = x;
        vertexList[10] = y;
        vertexList[11] = z + (sz * (iso - fieldNeighbours[0])) / (fieldNeighbours[3] - fieldNeighbours[0]);
    }

    if (bits & 16) {
        vertexList[12] = x + (sx * (iso - fieldNeighbours[4])) / (fieldNeighbours[5] - fieldNeighbours[4]);
        vertexList[13] = y + sy;
        vertexList[14] = z;
    }

    if (bits & 32) {
        vertexList[15] = x + sx;
        vertexList[16] = y + sy;
        vertexList[17] = z + (sz * (iso - fieldNeighbours[5])) / (fieldNeighbours[6] - fieldNeighbours[5]);
    }

    if (bits & 64) {
        vertexList[18] = x + (sx * (iso - fieldNeighbours[7])) / (fieldNeighbours[6] - fieldNeighbours[7]);
        vertexList[19] = y + sy;
        vertexList[20] = z + sz;
    }

    if (bits & 128) {
        vertexList[21] = x;
        vertexList[22] = y + sy;
        vertexList[23] = z + (sz * (iso - fieldNeighbours[4])) / (fieldNeighbours[7] - fieldNeighbours[4]);
    }

    if (bits & 256) {
        vertexList[24] = x;
        vertexList[25] = y + (sy * (iso - fieldNeighbours[0])) / (fieldNeighbours[4] - fieldNeighbours[0]);
        vertexList[26] = z;
    }

    if (bits & 512) {
        vertexList[27] = x + sx;
        vertexList[28] = y + (sy * (iso - fieldNeighbours[1])) / (fieldNeighbours[5] - fieldNeighbours[1]);
        vertexList[29] = z;
    }

    if (bits & 1024) {
        vertexList[30] = x + sx;
        vertexList[31] = y + (sy * (iso - fieldNeighbours[2])) / (fieldNeighbours[6] - fieldNeighbours[2]);
        vertexList[32] = z + sz;
    }

    if (bits & 2048) {
        vertexList[33] = x;
        vertexList[34] = y + (sy * (iso - fieldNeighbours[3])) / (fieldNeighbours[7] - fieldNeighbours[3]);
        vertexList[35] = z + sz;
    }

    index <<= 4;

    for (k = 0; triangleTable[index + k] !== -1; k++) {
        o = triangleTable[index + k] * 3;

        positions.push(vertexList[o]);
        positions.push(vertexList[o + 1]);
        positions.push(vertexList[o + 2]);
    }
};
