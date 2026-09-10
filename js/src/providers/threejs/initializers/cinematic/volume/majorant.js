// The medium's local majorant: for every macrocell of the volume, the largest extinction any
// point inside it can have. Delta tracking then steps at the density in front of the ray instead
// of the density of the box's single densest voxel, and a cell of air is crossed in one step
// rather than a thousand. The grid is built on the GPU because the volume lives there - a CT scan
// is hundreds of megabytes and a loop over it in JS would freeze the page for seconds.
const THREE = require('three');
const glsl = require('./glsl');

// voxels per macrocell edge
const CELL_SIZE = 8;
// covers the float32 logarithm in the shader against this float64 one, as the global bound does
const HEADROOM = 1.0 + 1e-5;
// the reduction reads one voxel past every face, because trilinear interpolation inside the cell
// does too: without it the majorant can come out below the extinction and the estimator is biased
const BORDER = 1;

const REDUCE_VERTEX = /* glsl */`
    void main() {
        gl_Position = vec4( position, 1.0 );
    }
`;

// one draw over an atlas of the grid's z layers: min in .r, max in .g, raw volume values
const REDUCE_FRAGMENT = /* glsl */`
    precision highp float;
    precision highp sampler3D;

    uniform sampler3D volumeTexture;
    uniform ivec3 voxels;
    uniform ivec3 cells;
    uniform ivec2 tiles;
    uniform int cellSize;
    uniform int border;

    void main() {
        ivec2 px = ivec2( gl_FragCoord.xy );
        ivec2 tile = px / cells.xy;
        int cellZ = tile.y * tiles.x + tile.x;

        // the atlas is a rectangle and the grid is not: the tail of the last tile row holds no cell
        if ( cellZ >= cells.z ) {
            gl_FragColor = vec4( 0.0 );
            return;
        }

        ivec3 cell = ivec3( px - tile * cells.xy, cellZ );
        ivec3 lo = max( cell * cellSize - border, ivec3( 0 ) );
        ivec3 hi = min( ( cell + 1 ) * cellSize + border, voxels ) - 1;

        float vmin = 1e30;
        float vmax = - 1e30;

        for ( int z = lo.z; z <= hi.z; z ++ ) {
            for ( int y = lo.y; y <= hi.y; y ++ ) {
                for ( int x = lo.x; x <= hi.x; x ++ ) {
                    float v = texelFetch( volumeTexture, ivec3( x, y, z ), 0 ).x;

                    // the raster reads NaN as no matter, and min() with one is undefined
                    if ( v == v ) {
                        vmin = min( vmin, v );
                        vmax = max( vmax, v );
                    }
                }
            }
        }

        gl_FragColor = vec4( vmin, vmax, 0.0, 1.0 );
    }
`;

// st[ k ] holds the maximum over every window of 2^k entries, so a range maximum is two lookups
function buildSparseMax(alpha) {
    const n = alpha.length;
    const levels = Math.max(1, Math.floor(Math.log2(n)) + 1);
    const st = [alpha];

    for (let k = 1; k < levels; k++) {
        const half = 1 << (k - 1);
        const prev = st[k - 1];
        const cur = new Float32Array(n);

        for (let i = 0; i + (1 << k) <= n; i++) {
            cur[i] = Math.max(prev[i], prev[i + half]);
        }

        st.push(cur);
    }

    return st;
}

// maximum over [ lo, hi ], both inclusive
function rangeMax(st, lo, hi) {
    const k = 31 - Math.clz32(hi - lo + 1);

    return Math.max(st[k][lo], st[k][hi - (1 << k) + 1]);
}

// The largest mean over any axis-aligned line of cells. A ray's tracking cost is the integral of
// the bound along it, so this bounds what an axis-aligned ray pays where the mean over all cells
// would understate it and the global bound would overstate it.
function worstLine(sigma, dims) {
    const [nx, ny, nz] = dims;
    let worst = 0;

    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            let sum = 0;

            for (let x = 0; x < nx; x++) {
                sum += sigma[(z * ny + y) * nx + x];
            }

            worst = Math.max(worst, sum / nx);
        }
    }

    for (let z = 0; z < nz; z++) {
        for (let x = 0; x < nx; x++) {
            let sum = 0;

            for (let y = 0; y < ny; y++) {
                sum += sigma[(z * ny + y) * nx + x];
            }

            worst = Math.max(worst, sum / ny);
        }
    }

    for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
            let sum = 0;

            for (let z = 0; z < nz; z++) {
                sum += sigma[(z * ny + y) * nx + x];
            }

            worst = Math.max(worst, sum / nz);
        }
    }

    return worst;
}

class MajorantGrid {
    constructor() {
        this.material = null;
        this.scene = null;
        this.camera = null;
        this.texture = null;
        this.cells = new THREE.Vector3(1, 1, 1);
        // uvw times this is the ray's coordinate in cells; not the cell count, because the
        // last cell reaches past the box when the voxel count is not a multiple of the edge
        this.scale = new THREE.Vector3(1, 1, 1);
        // the largest mean bound over any axis-aligned line through the grid: the step budget
        // is spent along one ray, so neither the mean over cells nor the global bound
        // describes what the worst ray pays
        this.worstLine = 0;
        // the reduction is the expensive half and only the volume's own data feeds it, so a
        // transfer function edit reuses it
        this.reduced = {
            source: null, version: -1, cellSize: 0, dims: null, minmax: null,
        };
        // syncVolume runs on every material edit, and re-uploading the grid on a roughness
        // drag would cost more than the drag saves
        this.built = '';
    }

    // the fullscreen triangle the reduction draws through
    prepare() {
        if (this.material !== null) {
            return;
        }

        const geometry = new THREE.BufferGeometry();

        const corners = new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]);

        geometry.setAttribute('position', new THREE.BufferAttribute(corners, 3));

        this.material = new THREE.ShaderMaterial({
            uniforms: {
                volumeTexture: { value: null },
                voxels: { value: new THREE.Vector3() },
                cells: { value: new THREE.Vector3() },
                tiles: { value: new THREE.Vector2() },
                cellSize: { value: CELL_SIZE },
                border: { value: BORDER },
            },
            vertexShader: REDUCE_VERTEX,
            fragmentShader: REDUCE_FRAGMENT,
            depthTest: false,
            depthWrite: false,
        });

        const mesh = new THREE.Mesh(geometry, this.material);

        mesh.frustumCulled = false;
        this.scene = new THREE.Scene();
        this.scene.add(mesh);
        // the triangle is already in clip space
        this.camera = new THREE.Camera();
    }

    // the per-cell min and max of the raw volume, or null when this context cannot render float
    reduce(renderer, source, cellSize) {
        const { image } = source;
        const voxels = [image.width, image.height, image.depth];

        if (!voxels.every((n) => n > 0)) {
            return null;
        }

        const cells = voxels.map((n) => Math.ceil(n / cellSize));
        const cached = this.reduced;

        if (cached.minmax !== null && cached.source === source
            && cached.version === source.version && cached.cellSize === cellSize) {
            return cached;
        }

        if (!renderer.extensions.has('EXT_color_buffer_float')) {
            return null;
        }

        const maxSize = renderer.capabilities.maxTextureSize;
        const tilesX = Math.max(1, Math.min(cells[2], Math.floor(maxSize / cells[0])));
        const tilesY = Math.ceil(cells[2] / tilesX);

        if (tilesY * cells[1] > maxSize) {
            return null;
        }

        this.prepare();

        const width = tilesX * cells[0];
        const height = tilesY * cells[1];
        const target = new THREE.WebGLRenderTarget(width, height, {
            format: THREE.RGBAFormat,
            type: THREE.FloatType,
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            depthBuffer: false,
            stencilBuffer: false,
            generateMipmaps: false,
        });
        const u = this.material.uniforms;

        u.volumeTexture.value = source;
        u.voxels.value.set(voxels[0], voxels[1], voxels[2]);
        u.cells.value.set(cells[0], cells[1], cells[2]);
        u.tiles.value.set(tilesX, tilesY);
        u.cellSize.value = cellSize;
        u.border.value = BORDER;

        const previous = renderer.getRenderTarget();

        renderer.setRenderTarget(target);
        renderer.render(this.scene, this.camera);
        renderer.setRenderTarget(previous);

        const atlas = new Float32Array(width * height * 4);

        renderer.readRenderTargetPixels(target, 0, 0, width, height, atlas);
        target.dispose();
        u.volumeTexture.value = null;

        const count = cells[0] * cells[1] * cells[2];
        const minmax = new Float32Array(count * 2);

        for (let z = 0; z < cells[2]; z++) {
            const tx = (z % tilesX) * cells[0];
            const ty = Math.floor(z / tilesX) * cells[1];

            for (let y = 0; y < cells[1]; y++) {
                const src = ((ty + y) * width + tx) * 4;
                const dst = ((z * cells[1] + y) * cells[0]) * 2;

                for (let x = 0; x < cells[0]; x++) {
                    minmax[dst + x * 2] = atlas[src + x * 4];
                    minmax[dst + x * 2 + 1] = atlas[src + x * 4 + 1];
                }
            }
        }

        this.reduced = {
            source, version: source.version, cellSize, dims: cells, minmax,
        };

        return this.reduced;
    }

    // The grid the shader samples, or a single cell holding the global majorant when the volume's
    // min-max pass is unavailable - one cell spanning the box makes the traversal degenerate into
    // the plain Woodcock loop, so the medium is correct either way.
    update({
        renderer, source, packedTF, tfSize, tfId, tfVersion, low, high, alphaCoef, sigmaMax,
        cellSize,
    }) {
        const size = cellSize || CELL_SIZE;
        const reduced = renderer && source && source.image
            ? this.reduce(renderer, source, size) : null;

        if (reduced === null) {
            return this.single(sigmaMax);
        }

        const voxels = [source.image.width, source.image.height, source.image.depth];
        // the ids are what make this a cache and not a collision: two volumes of the same
        // shape agree on every version number, so keying on values alone would hand the
        // second one the first one's bounds - and a bound from the wrong data is not a bound
        const key = [source.id, reduced.version, reduced.cellSize, tfId, tfVersion,
            low, high, alphaCoef].join('|');

        if (key === this.built && this.texture !== null) {
            return {
                texture: this.texture,
                cells: this.cells,
                scale: this.scale,
                worstLine: this.worstLine,
            };
        }

        this.built = key;

        const alpha = new Float32Array(tfSize);

        for (let i = 0; i < tfSize; i++) {
            alpha[i] = ((packedTF[i] >>> 24) & 0xff) / 255.0;
        }

        const st = buildSparseMax(alpha);
        const dims = reduced.dims;
        const count = dims[0] * dims[1] * dims[2];
        const sigma = new Float32Array(count);
        const range = high - low;
        const last = tfSize - 1;
        // the transfer function is read at min( scaled, 0.99 ), so nothing above it is reachable
        const top = 0.99;

        for (let i = 0; i < count; i++) {
            const vmin = reduced.minmax[i * 2];
            const vmax = reduced.minmax[i * 2 + 1];

            if (!(vmax >= vmin) || !(range !== 0)) {
                // every voxel NaN, or a colour range the raster itself cannot map
                sigma[i] = vmax >= vmin ? sigmaMax : 0.0;
            } else {
                let s0 = (vmin - low) / range;
                let s1 = (vmax - low) / range;

                if (s0 > s1) {
                    const swap = s0;

                    s0 = s1;
                    s1 = swap;
                }

                if (!(s1 > 0.0)) {
                    // the whole cell sits at or below low, where the raster draws nothing
                    sigma[i] = 0.0;
                } else {
                    const lo = Math.min(Math.max(Math.floor(Math.max(s0, 0.0) * tfSize), 0), last);
                    const hi = Math.min(Math.max(Math.floor(Math.min(s1, top) * tfSize), 0), last);
                    const a = rangeMax(st, Math.min(lo, hi), hi);

                    // the same headroom the global bound carries, for the same reason: the
                    // shader takes this logarithm in float32 and this one is float64
                    sigma[i] = a > 0.0 ? glsl.alphaToSigma(alphaCoef, a) * HEADROOM : 0.0;
                }
            }
        }

        return this.upload(sigma, dims, voxels.map((n) => n / size));
    }

    // the degenerate grid: one cell over the whole box, at the global majorant, which makes
    // the walk the plain Woodcock loop
    single(sigmaMax) {
        this.built = `single|${sigmaMax}`;
        return this.upload(new Float32Array([sigmaMax]), [1, 1, 1], [1, 1, 1]);
    }

    upload(sigma, dims, scale) {
        this.worstLine = worstLine(sigma, dims);
        const fits = this.texture !== null && this.texture.image.width === dims[0]
            && this.texture.image.height === dims[1] && this.texture.image.depth === dims[2];

        if (!fits) {
            if (this.texture !== null) {
                this.texture.dispose();
            }

            this.texture = new THREE.Data3DTexture(sigma, dims[0], dims[1], dims[2]);
            this.texture.format = THREE.RedFormat;
            this.texture.type = THREE.FloatType;
            this.texture.minFilter = THREE.NearestFilter;
            this.texture.magFilter = THREE.NearestFilter;
            this.texture.wrapS = THREE.ClampToEdgeWrapping;
            this.texture.wrapT = THREE.ClampToEdgeWrapping;
            this.texture.wrapR = THREE.ClampToEdgeWrapping;
            this.texture.generateMipmaps = false;
            this.texture.unpackAlignment = 1;
        } else {
            this.texture.image.data = sigma;
        }

        this.texture.needsUpdate = true;
        this.cells.set(dims[0], dims[1], dims[2]);
        this.scale.set(scale[0], scale[1], scale[2]);

        return {
            texture: this.texture,
            cells: this.cells,
            scale: this.scale,
            worstLine: this.worstLine,
        };
    }

    // the reduction holds the volume's own array through the texture it was built from, which
    // is the largest thing on the page; a plot that drops its volume must not keep it alive
    forget() {
        this.reduced = {
            source: null, version: -1, cellSize: 0, dims: null, minmax: null,
        };
        this.built = '';
    }

    dispose() {
        if (this.texture !== null) {
            this.texture.dispose();
            this.texture = null;
        }

        if (this.material !== null) {
            this.scene.children[0].geometry.dispose();
            this.material.dispose();
            this.material = null;
            this.scene = null;
            this.camera = null;
        }

        this.forget();
    }
}

module.exports = {
    MajorantGrid,
    CELL_SIZE,
};
