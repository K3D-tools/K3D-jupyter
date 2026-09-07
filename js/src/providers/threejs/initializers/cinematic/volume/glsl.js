// GLSL for the cinematic volume: K3D's Volume as a sampled medium inside the fog machinery of
// three-gpu-pathtracer. Every K3D-side symbol carries the k3dVolume prefix so nothing collides
// with an upstream chunk. The extinction mirrors Volume.fragment.glsl, so the raster and the
// tracer agree on how much matter there is - only the light transport differs.

// the raster LUT is a 1024x1 canvas; packed RGBA8 in a uvec4 array costs no texture unit
const TF_SIZE = 1024;
// alpha 1.0 has infinite extinction; the cap keeps the majorant finite (mean free path 1/500)
const ALPHA_MAX = 0.998;
// bounds the tracking loop where the majorant dwarfs the extinction along the ray
const MAX_STEPS = 2048;
// bounds the walk over the majorant grid; a ray crosses at most nx + ny + nz cells
const MAX_CELLS = 2048;
// the gradient hybrid: a collision is shaded as a surface with probability 1 - exp(-SURFACE_K * m),
// m the normalised intensity change over one gradient step; PHASE_G is the Henyey-Greenstein
// asymmetry of the gas events
const SURFACE_K = 8.0;
const PHASE_G = 0.85;

// extinction per unit of the box's local frame for a transfer function alpha, the CPU twin of
// k3dVolumeAlphaToSigma - the majorant has to be computed with the same formula
function alphaToSigma(alphaCoef, alpha) {
    return -alphaCoef * Math.log(1.0 - Math.min(alpha, ALPHA_MAX));
}

// uniforms, the renamed PCG generator and the medium helpers; goes before the upstream traceScene
const volumeDeclarations = /* glsl */`
    #define K3D_VOLUME_TF_SIZE ${TF_SIZE}
    #define K3D_VOLUME_ALPHA_MAX ${ALPHA_MAX}
    #define K3D_VOLUME_MAX_STEPS ${MAX_STEPS}
    #define K3D_VOLUME_MAX_CELLS ${MAX_CELLS}

    // set by traceScene when the medium scattered: the transfer function colour and the texture
    // coordinate of the collision, and whether the step cap ended the walk there
    bool k3dVolumeHit = false;
    bool k3dVolumeHitCapped = false;
    vec3 k3dVolumeHitAlbedo = vec3( 0.0 );
    vec3 k3dVolumeHitUvw = vec3( 0.0 );
    // set by getSurfaceRecord: the last event of the path happened inside the medium, so the
    // light it collects from here is the medium's to scale
    bool k3dVolumeLastEvent = false;
    // set by k3dVolumeClassify: this collision is shaded as a surface with this normal
    bool k3dVolumeHitSurface = false;
    vec3 k3dVolumeHitNormal = vec3( 0.0, 0.0, 1.0 );

    #if K3D_VOLUME

    uniform int volumeEnabled;
    uniform sampler3D volumeTexture;
    uniform uvec4 volumeTF[ K3D_VOLUME_TF_SIZE / 4 ];
    uniform int volumeTFSize;
    uniform float volumeLow;
    uniform float volumeHigh;
    uniform float volumeAlphaCoef;
    uniform float volumeSigmaMax;
    // the largest extinction each macrocell of the volume can hold
    uniform sampler3D volumeMajorant;
    uniform vec3 volumeMajorantCells;
    // uvw times this is the position in cells; a cell covers a fixed number of voxels, so
    // with a voxel count that is not a multiple of it the last cell reaches past the box
    uniform vec3 volumeMajorantScale;
    uniform mat4 volumeInvMatrix;
    uniform float volumeLightScale;
    uniform float volumeRoughness;
    uniform float volumeMetalness;
    uniform vec3 volumeGradientStep;
    uniform float volumeGradientScale;
    uniform float volumeSurfaceK;
    uniform float volumePhaseG;

    // copied from three-gpu-pathtracer 0.0.24 src/shader/rand/pcg.glsl.js with renamed symbols:
    // the upstream chunk is compiled out under RANDOM_TYPE 2 and its rng_initialize collides with
    // stratified.glsl.js, while a tracking loop of variable length needs an unindexed generator
    uvec4 k3dVolumeRngState;

    void k3dVolumeRngInit( vec2 p, int frame, uvec4 salt ) {
        k3dVolumeRngState = uvec4( p, uint( frame ), uint( p.x ) + uint( p.y ) ) ^ salt;
    }

    void k3dVolumePcg4d( inout uvec4 v ) {
        v = v * 1664525u + 1013904223u;
        v.x += v.y * v.w;
        v.y += v.z * v.x;
        v.z += v.x * v.y;
        v.w += v.y * v.z;
        v = v ^ ( v >> 16u );
        v.x += v.y * v.w;
        v.y += v.z * v.x;
        v.z += v.x * v.y;
        v.w += v.y * v.z;
    }

    // returns [ 0, 1 ]
    float k3dVolumeRand() {
        k3dVolumePcg4d( k3dVolumeRngState );
        return float( k3dVolumeRngState.x ) / float( 0xffffffffu );
    }

    // the LUT texel a NearestFilter texture() lookup at x would return
    vec4 k3dVolumeTF( float x ) {
        int i = clamp( int( x * float( volumeTFSize ) ), 0, volumeTFSize - 1 );
        uint p = volumeTF[ i >> 2 ][ i & 3 ];
        return vec4( p & 0xffu, ( p >> 8u ) & 0xffu, ( p >> 16u ) & 0xffu, ( p >> 24u ) & 0xffu ) / 255.0;
    }

    // Volume.fragment.glsl takes 1 - (1 - a)^(step * alpha_coef) per step, so the extinction per
    // unit of the box's local frame is -alpha_coef * ln(1 - a)
    float k3dVolumeAlphaToSigma( float a ) {
        return - volumeAlphaCoef * log( 1.0 - min( a, K3D_VOLUME_ALPHA_MAX ) );
    }

    // uvw is the box-local position + 0.5, the raster's texture coordinate
    float k3dVolumeExtinction( vec3 uvw, out vec3 albedo ) {
        float px = texture( volumeTexture, uvw ).x;
        float scaled = ( px - volumeLow ) / ( volumeHigh - volumeLow );
        albedo = vec3( 0.0 );

        // the raster skips samples at or below low; the negated test also drops NaN
        if ( ! ( scaled > 0.0 ) ) {
            return 0.0;
        }

        vec4 tf = k3dVolumeTF( min( scaled, 0.99 ) );
        albedo = tf.rgb;
        return k3dVolumeAlphaToSigma( tf.a );
    }

    // the exposure of the medium: 1 is the physics, and it multiplies only what light reaches
    // an event inside the medium, so occlusion keeps its shape and nothing else is touched
    float k3dVolumeLightScale() {
        return k3dVolumeLastEvent ? volumeLightScale : 1.0;
    }

    // the raster's normalised intensity at a texture coordinate, 0 at or below low
    float k3dVolumeScaled( vec3 uvw ) {
        float scaled = ( texture( volumeTexture, uvw ).x - volumeLow ) / ( volumeHigh - volumeLow );
        return scaled > 0.0 ? min( scaled, 1.0 ) : 0.0;
    }

    // Henyey-Greenstein, normalised over the sphere; cosTheta between propagation and the new direction
    float k3dVolumePhaseHG( float cosTheta ) {
        float g = volumePhaseG;
        float d = 1.0 + g * g - 2.0 * g * cosTheta;
        return ( 1.0 - g * g ) / ( 4.0 * PI * d * sqrt( d ) );
    }

    // Kroes' hybrid: the collision becomes a surface with probability 1 - exp(-k m), where m is
    // the intensity change per voxel - the step is the same distance in world units on every
    // axis, and volumeGradientScale turns the difference over it into that rate, so neither the
    // voxel shape, the box orientation, the grid resolution nor gradient_step move the decision.
    // Six fetches, and only on the camera path
    void k3dVolumeClassify( vec3 rayDirection ) {
        k3dVolumeHitSurface = false;

        if ( ! k3dVolumeHit || k3dVolumeHitCapped || ! ( volumeSurfaceK > 0.0 ) ) {
            return;
        }

        vec3 p = k3dVolumeHitUvw;
        vec3 h = volumeGradientStep;
        vec3 g = vec3(
            k3dVolumeScaled( p - vec3( h.x, 0.0, 0.0 ) ) - k3dVolumeScaled( p + vec3( h.x, 0.0, 0.0 ) ),
            k3dVolumeScaled( p - vec3( 0.0, h.y, 0.0 ) ) - k3dVolumeScaled( p + vec3( 0.0, h.y, 0.0 ) ),
            k3dVolumeScaled( p - vec3( 0.0, 0.0, h.z ) ) - k3dVolumeScaled( p + vec3( 0.0, 0.0, h.z ) )
        );
        float m = length( g ) * volumeGradientScale;

        // dimension 0 of the stratified table, which upstream reads only for the camera ray, at
        // bounce index 0; the table is bounces + 15 dimensions wide, so a new index would fall
        // outside it for the lowest cinematic_bounces and freeze this decision per pixel
        if ( rand( 0 ) >= 1.0 - exp( - volumeSurfaceK * m ) ) {
            return;
        }

        // g points towards falling density along the box axes; those axes in world space are the
        // normalised columns of the inverse transpose, so the world normal is their combination
        mat3 axes = transpose( mat3( volumeInvMatrix ) );
        vec3 n = normalize( axes[ 0 ] ) * g.x + normalize( axes[ 1 ] ) * g.y + normalize( axes[ 2 ] ) * g.z;

        if ( ! ( dot( n, n ) > 1e-20 ) ) {
            return;
        }

        n = normalize( n );
        k3dVolumeHitNormal = dot( n, rayDirection ) > 0.0 ? - n : n;
        k3dVolumeHitSurface = true;
    }

    // slab test against the unit box in its local frame; d is left unnormalised so that t is the
    // same parameter as in world space
    bool k3dVolumeBoxSpan( vec3 o, vec3 d, out float tEnter, out float tExit ) {
        vec3 inv = 1.0 / d;
        vec3 t0 = ( vec3( - 0.5 ) - o ) * inv;
        vec3 t1 = ( vec3( 0.5 ) - o ) * inv;

        // an axis the ray runs parallel to: outside the slab means no hit, inside means no bound
        bvec3 par = lessThan( abs( d ), vec3( 1e-12 ) );
        if ( any( par ) && any( greaterThan( abs( o ) * vec3( par ), vec3( 0.5 ) ) ) ) {
            return false;
        }
        if ( par.x ) { t0.x = - INFINITY; t1.x = INFINITY; }
        if ( par.y ) { t0.y = - INFINITY; t1.y = INFINITY; }
        if ( par.z ) { t0.z = - INFINITY; t1.z = INFINITY; }

        vec3 tn = min( t0, t1 );
        vec3 tf = max( t0, t1 );
        tEnter = max( max( tn.x, tn.y ), tn.z );
        tExit = min( min( tf.x, tf.y ), tf.z );

        return tExit > tEnter;
    }

    #else

    float k3dVolumeLightScale() {
        return 1.0;
    }

    void k3dVolumeClassify( vec3 rayDirection ) {
    }

    #endif
`;

// replaces traceScene: delta tracking through the sampled medium, surfaces from upstream
const volumeTraceScene = /* glsl */`
    int traceScene( Ray ray, Material fogMaterial, inout SurfaceHit surfaceHit ) {
        k3dVolumeHit = false;
        k3dVolumeHitCapped = false;

        #if K3D_VOLUME

        if ( volumeEnabled == 1 && fogMaterial.fogVolume ) {
            // the homogeneous fog of the boundary material is replaced by the medium, not added
            Material surfacesOnly = fogMaterial;
            surfacesOnly.fogVolume = false;
            int hitType = traceSceneUpstream( ray, surfacesOnly, surfaceHit );

            vec3 o = ( volumeInvMatrix * vec4( ray.origin, 1.0 ) ).xyz;
            vec3 d = ( volumeInvMatrix * vec4( ray.direction, 0.0 ) ).xyz;
            float tEnter;
            float tExit;

            // negated so a NaN majorant reads as no medium instead of a 2048-step loop
            if ( ! ( volumeSigmaMax > 0.0 ) || ! k3dVolumeBoxSpan( o, d, tEnter, tExit ) ) {
                return hitType;
            }

            // extinction and majorant are per local unit; dirLen converts them to per unit of t
            float dirLen = length( d );
            float tEnd = min( tExit, hitType == NO_HIT ? INFINITY : surfaceHit.dist );
            float t = max( tEnter, 0.0 );

            // the ray in cell coordinates: one unit is one macrocell and the grid spans the box
            vec3 gO = ( o + 0.5 ) * volumeMajorantScale;
            vec3 gD = d * volumeMajorantScale;
            ivec3 nCells = ivec3( volumeMajorantCells );
            ivec3 cell = clamp( ivec3( floor( gO + gD * t ) ), ivec3( 0 ), nCells - 1 );
            ivec3 stepDir = ivec3( 0 );
            // per axis: the t of the next cell boundary, and the t it takes to cross one whole cell
            vec3 tNext = vec3( INFINITY );
            vec3 tDelta = vec3( INFINITY );

            if ( abs( gD.x ) > 1e-12 ) {
                stepDir.x = gD.x > 0.0 ? 1 : - 1;
                tDelta.x = abs( 1.0 / gD.x );
                tNext.x = max( ( float( cell.x ) + max( float( stepDir.x ), 0.0 ) - gO.x ) / gD.x, t );
            }

            if ( abs( gD.y ) > 1e-12 ) {
                stepDir.y = gD.y > 0.0 ? 1 : - 1;
                tDelta.y = abs( 1.0 / gD.y );
                tNext.y = max( ( float( cell.y ) + max( float( stepDir.y ), 0.0 ) - gO.y ) / gD.y, t );
            }

            if ( abs( gD.z ) > 1e-12 ) {
                stepDir.z = gD.z > 0.0 ? 1 : - 1;
                tDelta.z = abs( 1.0 / gD.z );
                tNext.z = max( ( float( cell.z ) + max( float( stepDir.z ), 0.0 ) - gO.z ) / gD.z, t );
            }

            // Woodcock tracking at the majorant of the cell in front of the ray rather than of the
            // whole box, walking the grid with a 3D-DDA: an empty cell is crossed in one step, and
            // restarting the exponential at every boundary is exact because it is memoryless.
            // A ray that exhausts the step cap is absorbed where it stands
            bool escaped = false;
            int steps = 0;

            for ( int c = 0; c < K3D_VOLUME_MAX_CELLS; c ++ ) {
                float tCell = min( min( tNext.x, tNext.y ), tNext.z );
                float tSeg = min( tCell, tEnd );
                float sigmaMax = texelFetch( volumeMajorant, cell, 0 ).x * dirLen;
                // the offset guards the far end of the medium, where a collision would land on
                // the surface the ray already hit; a cell boundary is not a surface, and skipping
                // it there would drop a sliver of the medium at every crossing
                float tStop = tSeg < tEnd ? tSeg : tEnd - RAY_OFFSET;

                if ( sigmaMax > 0.0 ) {
                    for ( ; steps < K3D_VOLUME_MAX_STEPS; steps ++ ) {
                        t -= log( max( k3dVolumeRand(), 1e-8 ) ) / sigmaMax;

                        if ( t >= tStop ) {
                            break;
                        }

                        vec3 albedo;
                        float sigma = k3dVolumeExtinction( o + d * t + 0.5, albedo ) * dirLen;

                        if ( k3dVolumeRand() * sigmaMax < sigma ) {
                            surfaceHit.side = 1.0;
                            surfaceHit.faceNormal = normalize( - ray.direction );
                            surfaceHit.dist = t;
                            k3dVolumeHit = true;
                            k3dVolumeHitAlbedo = albedo;
                            k3dVolumeHitUvw = o + d * t + 0.5;
                            return FOG_HIT;
                        }
                    }

                    if ( steps >= K3D_VOLUME_MAX_STEPS ) {
                        break;
                    }
                }

                if ( tSeg >= tEnd ) {
                    escaped = true;
                    break;
                }

                t = tCell;

                if ( tCell == tNext.x ) {
                    cell.x += stepDir.x;
                    tNext.x += tDelta.x;
                } else if ( tCell == tNext.y ) {
                    cell.y += stepDir.y;
                    tNext.y += tDelta.y;
                } else {
                    cell.z += stepDir.z;
                    tNext.z += tDelta.z;
                }

                if ( any( lessThan( cell, ivec3( 0 ) ) )
                    || any( greaterThanEqual( cell, nCells ) ) ) {
                    escaped = true;
                    break;
                }
            }

            if ( ! escaped ) {
                surfaceHit.side = 1.0;
                surfaceHit.faceNormal = normalize( - ray.direction );
                surfaceHit.dist = t;
                k3dVolumeHit = true;
                k3dVolumeHitAlbedo = vec3( 0.0 );
                k3dVolumeHitCapped = true;
                return FOG_HIT;
            }

            return hitType;
        }

        #endif

        return traceSceneUpstream( ray, fogMaterial, surfaceHit );
    }
`;

// replaces getSurfaceRecord: a gas event takes the transfer function colour of its point, a
// surface event (k3dVolumeClassify) gets the record upstream would build for a rough dielectric
const volumeSurfaceRecord = /* glsl */`
    #if K3D_VOLUME

    void k3dVolumeSurfaceRecord( float accumulatedRoughness, inout SurfaceRecord surf ) {
        vec3 n = k3dVolumeHitNormal;

        surf.volumeParticle = false;
        surf.faceNormal = n;
        surf.frontFace = true;
        surf.normal = n;
        surf.normalBasis = getBasisFromNormal( n );
        surf.normalInvBasis = inverse( surf.normalBasis );
        surf.ior = 1.5;
        surf.eta = 1.0 / surf.ior;
        surf.f0 = iorRatioToF0( surf.eta );
        surf.roughness = volumeRoughness * volumeRoughness;
        surf.filteredRoughness = applyFilteredGlossy( surf.roughness, accumulatedRoughness );
        surf.metalness = volumeMetalness;
        surf.color = k3dVolumeHitAlbedo;
        surf.emission = vec3( 0.0 );
        surf.transmission = 0.0;
        surf.thinFilm = false;
        surf.attenuationColor = vec3( 1.0 );
        surf.attenuationDistance = INFINITY;
        surf.clearcoatNormal = n;
        surf.clearcoatBasis = surf.normalBasis;
        surf.clearcoatInvBasis = surf.normalInvBasis;
        surf.clearcoat = 0.0;
        surf.clearcoatRoughness = 0.0;
        surf.filteredClearcoatRoughness = 0.0;
        surf.sheen = 0.0;
        surf.sheenColor = vec3( 0.0 );
        surf.sheenRoughness = 0.0;
        surf.iridescence = 0.0;
        surf.iridescenceIor = 1.3;
        surf.iridescenceThickness = 0.0;
        surf.specularColor = vec3( 1.0 );
        surf.specularIntensity = 1.0;
    }

    #endif

    int getSurfaceRecord(
        Material material, SurfaceHit surfaceHit, sampler2DArray attributesArray,
        float accumulatedRoughness,
        inout SurfaceRecord surf
    ) {
        #if K3D_VOLUME

        bool medium = material.fogVolume && k3dVolumeHit;

        if ( medium ) {
            k3dVolumeLastEvent = true;

            if ( k3dVolumeHitSurface ) {
                k3dVolumeSurfaceRecord( accumulatedRoughness, surf );
                return HIT_SURFACE;
            }

            material.color = k3dVolumeHitAlbedo;
        }

        #endif

        int hit = getSurfaceRecordUpstream( material, surfaceHit, attributesArray, accumulatedRoughness, surf );

        #if K3D_VOLUME

        // a surface upstream discarded (stochastic transparency, alpha test, sidedness) is not
        // an event: main() steps past it with the same ray, still leaving the medium event
        if ( hit != SKIP_SURFACE ) {
            k3dVolumeLastEvent = medium;
        }

        #endif

        // upstream's fog branch leaves frontFace unset; read as false, main() would attenuate the
        // throughput over the collision distance by an unset medium colour and end the path there
        if ( surf.volumeParticle ) {
            surf.frontFace = true;
        }

        return hit;
    }
`;

// wraps directLightContribution: light collected at an event inside the medium is the
// medium's to expose
const volumeDirectLight = /* glsl */`
    vec3 directLightContribution( vec3 worldWo, SurfaceRecord surf, RenderState state, vec3 rayOrigin ) {
        return directLightContributionUpstream( worldWo, surf, state, rayOrigin ) * k3dVolumeLightScale();
    }
`;

// wraps bsdfSample / bsdfResult: Henyey-Greenstein for the gas events; |g| below 1e-3 keeps
// upstream's isotropic branch, so g = 0 is the stage-1 shader
const volumeBsdf = /* glsl */`
    float bsdfResult( vec3 worldWo, vec3 worldWi, SurfaceRecord surf, inout vec3 color ) {
        #if K3D_VOLUME

        if ( surf.volumeParticle && abs( volumePhaseG ) >= 1e-3 ) {
            float pdf = k3dVolumePhaseHG( dot( - worldWo, worldWi ) );
            color = surf.color * pdf;
            return pdf;
        }

        #endif

        return bsdfResultUpstream( worldWo, worldWi, surf, color );
    }

    ScatterRecord bsdfSample( vec3 worldWo, SurfaceRecord surf ) {
        #if K3D_VOLUME

        if ( surf.volumeParticle && abs( volumePhaseG ) >= 1e-3 ) {
            // PBRT's inversion around the propagation direction - worldWo; u = 1 goes straight on
            vec2 r = rand2( 16 );
            float g = volumePhaseG;
            float sqrTerm = ( 1.0 - g * g ) / ( 1.0 - g + 2.0 * g * r.x );
            float cosTheta = clamp( ( 1.0 + g * g - sqrTerm * sqrTerm ) / ( 2.0 * g ), - 1.0, 1.0 );
            float sinTheta = sqrt( max( 0.0, 1.0 - cosTheta * cosTheta ) );
            float phi = 2.0 * PI * r.y;

            ScatterRecord rec;
            rec.specularPdf = 0.0;
            rec.pdf = k3dVolumePhaseHG( cosTheta );
            rec.direction = normalize(
                getBasisFromNormal( - worldWo ) * vec3( sinTheta * cos( phi ), sinTheta * sin( phi ), cosTheta )
            );
            rec.color = surf.color * rec.pdf;

            return rec;
        }

        #endif

        return bsdfSampleUpstream( worldWo, surf );
    }
`;

// appended to the RNG setup in main(); pixelSeed is the stratified sampler's per-pixel offset,
// which K3D reseeds from cinematic_seed, so the medium is repeatable exactly when the rest is
const volumeRngInit = /* glsl */`
    #if K3D_VOLUME
    k3dVolumeRngInit( gl_FragCoord.xy, seed, uvec4( pixelSeed * 16777215.0 ) );
    #endif
`;

// K3D never proxies a light object into the traced scene - the environment map is the only
// emitter - so both light entry points are dead code holding two texture units the medium
// needs. A compile-time zero count keeps every estimator weight as it is today.
const volumeNoLights = /* glsl */`
    struct K3DLightsInfo { uint count; };
    K3DLightsInfo lights = K3DLightsInfo( 0u );
`;

// inserted after the light sampling chunk, so its own declarations stay intact. Only the forward
// hit in main() needs a stub: the branch that samples a light is cut out by the preprocessor and
// randomLightSample is left with no caller
const volumeNoLightMacros = /* glsl */`
    #define intersectLightAtIndex( lightsTex, rayOrigin, rayDirection, l, lightRec ) ( false )
`;

module.exports = {
    TF_SIZE,
    ALPHA_MAX,
    MAX_STEPS,
    MAX_CELLS,
    alphaToSigma,
    volumeDeclarations,
    volumeTraceScene,
    volumeSurfaceRecord,
    volumeBsdf,
    volumeDirectLight,
    volumeRngInit,
    volumeNoLights,
    volumeNoLightMacros,
    SURFACE_K,
    PHASE_G,
};
