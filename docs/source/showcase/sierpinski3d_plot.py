import numpy as np
import k3d


def generate():
    plot = k3d.plot(screenshot_scale=1.0)

    iteration = 4
    size = 3 ** iteration

    voxels = np.ones((size, size, size))

    def iterate(length, x, y, z):

        nl = length // 3

        if nl < 1:
            return

        margin = (nl - 1) // 2

        voxels[z - margin:z + margin + 1, y - margin:y + margin + 1, :] = 0
        voxels[z - margin:z + margin + 1, :, x - margin:x + margin + 1] = 0
        voxels[:, y - margin:y + margin + 1, x - margin:x + margin + 1] = 0

        for ix, iy, iz in np.ndindex((3, 3, 3)):
            if (1 if ix != 1 else 0) + (1 if iy != 1 else 0) + (1 if iz != 1 else 0) != 2:
                iterate(nl, x + (ix - 1) * nl, y + (iy - 1) * nl, z + (iz - 1) * nl)

    iterate(size, size // 2, size // 2, size // 2)

    plot += k3d.voxels(voxels.astype(np.uint8), color_map=(0xffff00))

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
