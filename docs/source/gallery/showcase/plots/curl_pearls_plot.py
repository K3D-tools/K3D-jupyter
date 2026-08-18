import numpy as np

import k3d


def build_scene():
    rng = np.random.default_rng(7)
    m = 72
    L = 1.25
    g = np.linspace(-L, L, m, dtype=np.float32)
    dg = g[1] - g[0]
    Zg, Yg, Xg = np.meshgrid(g, g, g, indexing='ij')
    r_grid = np.sqrt(Xg ** 2 + Yg ** 2 + Zg ** 2)

    # laminar by construction: a few long-wavelength sinusoids as the vector
    # potential, plus medium waves for turbulence. The radial envelope zeroes the
    # potential at the sphere, so streamlines curve along it instead of escaping.
    envelope = np.clip(1.0 - (r_grid / 1.02) ** 2, 0.0, 1.0).astype(np.float32)

    def low_freq(rng):
        f = np.zeros((m, m, m), np.float32)
        for sigma, amp in ((1.2, 1.0), (1.2, 1.0), (1.2, 1.0),
                           (3.0, 0.85), (3.0, 0.85), (5.0, 0.45)):
            k = rng.normal(0.0, sigma, 3).astype(np.float32)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            f += amp * rng.uniform(0.6, 1.0) * np.sin(Xg * k[0] + Yg * k[1] + Zg * k[2] + phase)
        return f.astype(np.float32)

    A = [low_freq(rng) * envelope for _ in range(3)]
    curl = np.stack([
        (np.roll(A[2], -1, 1) - np.roll(A[2], 1, 1)) - (np.roll(A[1], -1, 0) - np.roll(A[1], 1, 0)),
        (np.roll(A[0], -1, 0) - np.roll(A[0], 1, 0)) - (np.roll(A[2], -1, 2) - np.roll(A[2], 1, 2)),
        (np.roll(A[1], -1, 2) - np.roll(A[1], 1, 2)) - (np.roll(A[0], -1, 1) - np.roll(A[0], 1, 1)),
    ], axis=-1)

    def field_at(p):
        f = (p[:, ::-1] + L) / dg
        i0 = np.clip(np.floor(f).astype(np.int32), 0, m - 2)
        d = (f - i0).astype(np.float32)
        out = np.zeros((len(p), 3), np.float32)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = (np.abs(1 - dz - d[:, 0]) * np.abs(1 - dy - d[:, 1])
                         * np.abs(1 - dx - d[:, 2]))[:, None]
                    out += w * curl[i0[:, 0] + dz, i0[:, 1] + dy, i0[:, 2] + dx]
        return out

    S = 18             # seed nests
    T = 460            # beads per strand
    bead = 0.024       # base bead radius

    # a ribbon is a line of seeds carried by the flow - not a gaussian puff
    nests = rng.normal(size=(S * 3, 3)).astype(np.float32)
    nests = nests[np.linalg.norm(nests, axis=1) > 1e-3][:S]
    nests = nests / np.linalg.norm(nests, axis=1, keepdims=True)
    nests *= rng.uniform(0.1, 0.65, (S, 1)).astype(np.float32) ** 0.7
    per_nest = 9
    span = rng.normal(size=(S, 3)).astype(np.float32)
    span /= np.linalg.norm(span, axis=1, keepdims=True)
    offsets = np.linspace(-0.14, 0.14, per_nest, dtype=np.float32)
    seeds = (nests[:, None, :] + span[:, None, :] * offsets[None, :, None]
             ).reshape(-1, 3).astype(np.float32)
    S = len(seeds)

    pos = seeds.copy()
    alive = np.ones(S, bool)
    points = []
    sizes = []
    colors = []
    strand_scale = rng.uniform(0.45, 1.15, S).astype(np.float32)
    shades = np.array([0xF2F5FA, 0xE3E9F3, 0xD3DCEB], dtype=np.uint32)
    strand_shade = shades[rng.integers(0, 3, S)]

    for t in range(T):
        v = field_at(pos)
        speed = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        pos = pos + v / speed * (bead * 0.8)          # dense caterpillar spacing
        r = np.linalg.norm(pos, axis=1)
        alive &= (r < 1.03) & (speed[:, 0] > 5e-5)    # stagnation at the envelope
        if not alive.any():
            break
        taper = (1.0 - 0.65 * t / T)
        points.append(pos[alive].copy())
        sizes.append(np.full(alive.sum(), 2.0 * bead * taper, np.float32) * strand_scale[alive])
        colors.append(strand_shade[alive].copy())

    positions = np.concatenate(points).astype(np.float32)
    point_sizes = np.concatenate(sizes).astype(np.float32)
    point_colors = np.concatenate(colors).astype(np.uint32)

    plot = k3d.plot(grid_visible=False, camera_auto_fit=False,
                    background_color=0x08090B,
                    renderer='advanced', environment='neutral',
                    lighting=2.1, ao_radius=0.03, ao_strength=2.4,
                    screenshot_scale=1.0)

    # mesh spheres, not the 3d impostors - the beads must cast and receive
    # the ambient occlusion that sculpts the crevices between strands
    plot += k3d.points(positions, point_sizes=point_sizes, shader='mesh',
                       mesh_detail=2, colors=point_colors, roughness=0.35,
                       compression_level=7)
    plot.camera = [1.75, -1.75, 1.1, 0, 0, 0, 0, 0, 1]

    return plot


def generate():
    plot = build_scene()

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
