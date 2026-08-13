const THREE = require('three');

// BRDF_GGX and its two helpers live in lights_physical_pars_fragment - the chunk that
// assumes the whole mesh lighting pipeline - not in <bsdfs>. Sliced verbatim at require
// time, so three stays the single source of truth: after an upgrade a failed slice or a
// changed signature fails shader compilation loudly in the suite. F_Schlick and
// BRDF_Lambert come from <common>.
function sliceFunction(source, signature) {
    const start = source.indexOf(signature);

    if (start === -1) {
        throw new Error(`K3D: "${signature}" not found in three's lights_physical_pars_fragment`);
    }

    let depth = 0;
    let i = source.indexOf('{', start);

    for (; i < source.length; i++) {
        if (source[i] === '{') {
            depth++;
        } else if (source[i] === '}') {
            depth--;

            if (depth === 0) {
                break;
            }
        }
    }

    return source.slice(start, i + 1);
}

const chunk = ['float D_GGX(', 'float V_GGX_SmithCorrelated(', 'vec3 BRDF_GGX(']
    .map((signature) => sliceFunction(THREE.ShaderChunk.lights_physical_pars_fragment, signature))
    .join('\n');

module.exports = function injectGGX(fragmentShader) {
    return fragmentShader.replace('// K3D_GGX_CHUNK', chunk);
};
