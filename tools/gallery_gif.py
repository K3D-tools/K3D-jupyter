"""Render the streamlines showcase as an orbiting frame sequence, for the README and the
anywidget community gallery.

Frames only - the GIF is assembled with ffmpeg afterwards, because the docker image that owns the
pinned Chrome has no ffmpeg and the host has no pinned Chrome:

    docker compose run --rm k3d-build bash -lc \
        "cd /opt/app/src && python tools/gallery_gif.py --out /opt/app/src/.gifframes"

    ffmpeg -framerate 20 -i .gifframes/f%03d.png \
        -vf "split[a][b];[a]palettegen=max_colors=256:stats_mode=diff[p];\
[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle" -loop 0 imgs/streamlines.gif

The scene is the one from docs/source/gallery/showcase/plots/streamlines_plot.py, camera included.
The orbit is phased so the MIDDLE frame lands on that hand-picked camera: the anywidget gallery
extracts its still with sharp(bytes, {pages: 1, page: floor(npages / 2)}), so the midpoint is the
frame that ends up representing K3D in the grid.
"""
import argparse
import math
import os
import sys

# the same shape as k3d/test/conftest.py: put the checkout on the path so this runs against the
# working tree rather than whatever is installed, and needs no PYTHONPATH from the caller
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import k3d
from k3d.headless import get_headless_driver, k3d_remote

HERE = os.path.abspath(os.path.dirname(__file__))
ASSET = os.path.join(HERE, '..', 'docs', 'source', 'gallery', 'showcase', 'assets',
                     'streamlines_data.npz')

# straight from streamlines_plot.py - do not drift from the showcase
CAMERA = [0.0705, 0.0411, 0.0538,
          0.0511, 0.0391, 0.0493,
          -0.0798, 0.9872, 0.1265]


def build_plot(mesh_opacity=0.25):
    data = np.load(os.path.normpath(ASSET))

    streamlines = k3d.line(data['lines'],
                           width=0.00007,
                           attribute=data['v'],
                           color_map=k3d.matplotlib_color_maps.Inferno,
                           color_range=[0, 0.5],
                           shader='mesh')

    mesh = k3d.mesh(data['vertices'], data['indices'],
                    opacity=mesh_opacity,
                    wireframe=True,
                    color=0x0002)

    # camera_auto_fit off, or the first sync refits and discards the camera we set - which is
    # invisible in a long run (every later frame sets it again) and ruins exactly frame 0
    plot = k3d.plot(grid_visible=False, screenshot_scale=1.0, axes_helper=0,
                    menu_visibility=False, camera_auto_fit=False, grid_auto_fit=False)
    plot += streamlines
    plot += mesh
    plot.camera = CAMERA

    return plot, data


def orbit(centre, frame, frames, elevation, azimuth0):
    """A direction to look from - the distance is not ours to pick.

    setCameraToFitScene (Camera.js:57) keeps the camera's direction and recomputes only the
    position and the target, at radius * factor / sin(fov / 2) from the bounding sphere. So this
    returns a unit offset and camera_reset does the framing; the sin(fov / 2) term is exactly
    what a hand-rolled distance gets wrong, and at the default 60 deg fov it is a factor of two.

    Y-up, like the showcase camera, and not get_auto_camera: that one hardcodes up = [0, 0, 1]
    and measures pitch from +Z, so on this dataset a small pitch aims the camera almost along
    its own up vector and the frame comes out degenerate or empty.

    The phase puts azimuth0 on frame frames // 2 - the frame the anywidget gallery extracts as
    its still with sharp(bytes, {pages: 1, page: floor(npages / 2)}).
    """
    azimuth = math.radians(azimuth0 + 360.0 * (frame - frames // 2) / frames)
    elevation = math.radians(elevation)

    offset = np.array([math.cos(elevation) * math.cos(azimuth),
                       math.sin(elevation),
                       math.cos(elevation) * math.sin(azimuth)])

    return [*(centre + offset), *centre, 0.0, 1.0, 0.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='.gifframes', help='directory for the PNG frames')
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=450)
    parser.add_argument('--mesh-opacity', type=float, default=0.25,
                        help='the wireframe reads as a dark net once the card shrinks it')
    parser.add_argument('--factor', type=float, default=1.5,
                        help='camera_reset framing factor; higher pulls back')
    parser.add_argument('--elevation', type=float, default=12.0,
                        help='degrees above the XZ plane (the showcase camera sits at 5.7)')
    parser.add_argument('--azimuth', type=float, default=13.0,
                        help='degrees from +X towards +Z, for the middle frame '
                             '(the showcase camera sits at 13.1)')
    parser.add_argument('--only-frame', type=int, default=None,
                        help='render one frame index only, for probing a look')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    plot, _ = build_plot(args.mesh_opacity)
    headless = k3d_remote(plot, get_headless_driver(), width=args.width, height=args.height)

    # the visual suite's own determinism switch: without it the dithering jitter differs per run
    headless.browser.execute_script('window.randomMul = 0.0;')

    try:
        wanted = ([args.only_frame] if args.only_frame is not None
                  else list(range(args.frames)))

        # get_auto_grid, not a bounding box of our own: it is what the browser fits to, and it
        # copes with the NaN separators that divide the streamlines
        grid = np.array(plot.get_auto_grid(), dtype=np.float64)
        centre = (grid[:3] + grid[3:]) / 2.0

        for frame in wanted:
            plot.camera = orbit(centre, frame, args.frames, args.elevation, args.azimuth)
            headless.sync(hold_until_refreshed=True)

            # the direction is ours, the distance is K3D's - see orbit()
            headless.camera_reset(args.factor)

            # only_canvas: the DOM overlay carries the colour legend and any grid labels, and a
            # screenshot that includes it also includes whatever chrome the page has
            png = headless.get_screenshot(only_canvas=True)

            with open(os.path.join(args.out, 'f%03d.png' % frame), 'wb') as handle:
                handle.write(png)

            print('  %3d/%d' % (frame + 1, args.frames), end='\r', flush=True)
    finally:
        headless.close()

    print('\n%d frames in %s (middle frame f%03d is the gallery still)'
          % (args.frames, args.out, args.frames // 2))


if __name__ == '__main__':
    main()
