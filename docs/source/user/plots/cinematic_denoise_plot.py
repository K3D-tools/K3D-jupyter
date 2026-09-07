import numpy as np

import k3d


def generate():
    # A volume with structure at two scales: a smooth falloff a filter should leave alone, and a
    # ripple fine enough that an over-eager one would iron it flat. That is the pair the setting
    # has to separate, and a plain blob would not show the difference.
    g = np.linspace(-1, 1, 64, dtype=np.float32)
    z, y, x = np.meshgrid(g, g, g, indexing='ij')
    radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    blob = (np.exp(-(radius ** 2) / 0.35)
            * (0.75 + 0.25 * np.cos(18.0 * x))
            * 900).astype(np.float32)

    plot = k3d.plot(grid_visible=False,
                    camera_auto_fit=False,
                    screenshot_scale=1.0,
                    colorbar_object_id=0,
                    renderer='cinematic',
                    environment='studio')

    plot += k3d.volume(blob, samples=256, alpha_coef=15,
                       color_map=k3d.matplotlib_color_maps.jet,
                       color_range=[80, 900],
                       compression_level=7)

    # A budget this small leaves obvious grain, which is the point: the filter is worth about
    # four times the samples on a volume, so 24 filtered sit where roughly a hundred would.
    plot.cinematic_samples = 24
    plot.cinematic_bounces = 4
    plot.cinematic_denoise = 2.0

    plot.camera = [2.4, -2.4, 1.6, 0, 0, 0, 0, 0, 1]

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
