//jshint maxstatements:false, maxcomplexity:false, bitwise:false
//Performance reason

'use strict';

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

var vertexList = new Float32Array(12 * 3),
    fieldNeighbours = new Float32Array(8),
    edgeTable = require('./marchingCubesLookupTables').edgeTable,
    triangleTable = require('./marchingCubesLookupTables').triangleTable;

module.exports = function (positions, field, width, height, length, iso, x, y, z) {
    var index = 0,
        ptr = z * width * height + y * width + x,
        o = 0,
        bits,
        k;

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

    bits = edgeTable[index];

    if (bits === 0) {
        return;
    }

    if (bits & 1) {
        vertexList[0] = x + (iso - fieldNeighbours[0]) / (fieldNeighbours[1] - fieldNeighbours[0]);
        vertexList[1] = y;
        vertexList[2] = z;
    }

    if (bits & 2) {
        vertexList[3] = x + 1;
        vertexList[4] = y;
        vertexList[5] = z + (iso - fieldNeighbours[1]) / (fieldNeighbours[2] - fieldNeighbours[1]);
    }

    if (bits & 4) {
        vertexList[6] = x + (iso - fieldNeighbours[3]) / (fieldNeighbours[2] - fieldNeighbours[3]);
        vertexList[7] = y;
        vertexList[8] = z + 1;
    }

    if (bits & 8) {
        vertexList[9] = x;
        vertexList[10] = y;
        vertexList[11] = z + (iso - fieldNeighbours[0]) / (fieldNeighbours[3] - fieldNeighbours[0]);
    }

    if (bits & 16) {
        vertexList[12] = x + (iso - fieldNeighbours[4]) / (fieldNeighbours[5] - fieldNeighbours[4]);
        vertexList[13] = y + 1;
        vertexList[14] = z;
    }

    if (bits & 32) {
        vertexList[15] = x + 1;
        vertexList[16] = y + 1;
        vertexList[17] = z + (iso - fieldNeighbours[5]) / (fieldNeighbours[6] - fieldNeighbours[5]);
    }

    if (bits & 64) {
        vertexList[18] = x + (iso - fieldNeighbours[7]) / (fieldNeighbours[6] - fieldNeighbours[7]);
        vertexList[19] = y + 1;
        vertexList[20] = z + 1;
    }

    if (bits & 128) {
        vertexList[21] = x;
        vertexList[22] = y + 1;
        vertexList[23] = z + (iso - fieldNeighbours[4]) / (fieldNeighbours[7] - fieldNeighbours[4]);
    }

    if (bits & 256) {
        vertexList[24] = x;
        vertexList[25] = y + (iso - fieldNeighbours[0]) / (fieldNeighbours[4] - fieldNeighbours[0]);
        vertexList[26] = z;
    }

    if (bits & 512) {
        vertexList[27] = x + 1;
        vertexList[28] = y + (iso - fieldNeighbours[1]) / (fieldNeighbours[5] - fieldNeighbours[1]);
        vertexList[29] = z;
    }

    if (bits & 1024) {
        vertexList[30] = x + 1;
        vertexList[31] = y + (iso - fieldNeighbours[2]) / (fieldNeighbours[6] - fieldNeighbours[2]);
        vertexList[32] = z + 1;
    }

    if (bits & 2048) {
        vertexList[33] = x;
        vertexList[34] = y + (iso - fieldNeighbours[3]) / (fieldNeighbours[7] - fieldNeighbours[3]);
        vertexList[35] = z + 1;
    }

    index <<= 4;

    for (k = 0; triangleTable[index + k] !== -1; k++) {
        o = triangleTable[index + k] * 3;

        positions.push(vertexList[o] / (width - 1));
        positions.push(vertexList[o + 1] / (height - 1));
        positions.push(vertexList[o + 2] / (length - 1));
    }
};
