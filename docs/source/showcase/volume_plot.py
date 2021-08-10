import numpy as np
import k3d
import SimpleITK as sitk
import pathlib

path = pathlib.Path(__file__).parent.resolve()


def generate():
    plot = k3d.plot(screenshot_scale=1.0, camera_auto_fit=False)

    im_sitk = sitk.ReadImage(str(path) + '/assets/heart.mhd')
    img = sitk.GetArrayFromImage(im_sitk)
    size = np.array(im_sitk.GetSize()) * np.array(im_sitk.GetSpacing())
    im_sitk.GetSize()

    volume = k3d.volume(
        img.astype(np.float32),
        alpha_coef=1000,
        shadow='dynamic',
        samples=600,
        shadow_res=128,
        shadow_delay=250,
        color_range=[150, 750],
        color_map=(np.array(k3d.colormaps.matplotlib_color_maps.Gist_heat).reshape(-1, 4)
                   * np.array([1, 1.75, 1.75, 1.75])).astype(np.float32)
    )

    volume.transform.bounds = [-size[0] / 2, size[0] / 2,
                               -size[1] / 2, size[1] / 2,
                               -size[2] / 2, size[2] / 2]
    plot += volume

    plot.camera = [113.68958627222906, -154.65922433874593, 133.0600200640414, 3.812331338407114,
                   -6.061605333107961, -11.72517048593912, -0.3295525799943104, 0.46784677029710225,
                   0.8200698119926529]

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
