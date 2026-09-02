"""Compose the social card served as docs/source/_static/og-image.png.

Two steps, because rendering needs the pinned Chrome in the docker image and compositing does
not. First a single dark frame of the streamlines showcase, enlarged so it bleeds off the edges:

    docker compose run --rm k3d-build bash -lc \
        "cd /opt/app/src && python tools/gallery_gif.py --out /opt/app/src/.ogframe \
         --only-frame 30 --width 1200 --height 630 --background 0x000000 \
         --mesh-color 0x555C6B --mesh-opacity 0.35 --factor 0.32 --azimuth 93"

Then the card:

    python tools/og_card.py --frame .ogframe/f030.png --logo-dx 300

The logo is placed with a feathered alpha edge rather than a hard border, which works because
its own background is near-black (#0C0D11): the streamlines inside the logo run into the
rendered ones and the square stops reading as a pasted-on box.
"""
import argparse
import os

import numpy as np
from PIL import Image

HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_LOGO = os.path.join(HERE, '..', 'docs', 'source', '_static', 'logo.png')
DEFAULT_OUT = os.path.join(HERE, '..', 'docs', 'source', '_static', 'og-image.png')

# what Facebook, LinkedIn and Slack scale without cropping
SIZE = (1200, 630)


def feather_mask(size, edge):
    """Alpha that is opaque in the middle and falls to zero over `edge` pixels of the border."""
    run = np.minimum(np.arange(size), np.arange(size)[::-1]).astype(np.float32)
    dist = np.minimum(run[:, None], run[None, :])
    t = np.clip(dist / edge, 0.0, 1.0)

    # smoothstep, not linear: a linear ramp leaves a visible seam where it meets full opacity
    return Image.fromarray((t * t * (3.0 - 2.0 * t) * 255.0).astype(np.uint8), 'L')


def background(frame, size, recentre, threshold=30):
    """The render as the card's ground, optionally shifted onto its own lit pixels.

    Recentring only helps a loose framing, where camera_reset fits the bounding sphere and a
    tall object leaves a third of the card empty. At a tight --factor the render already fills
    the frame and shifting it just opens a black band on one side, so this is off by default.
    """
    src = Image.open(frame).convert('RGB')
    card = Image.new('RGB', size, 'black')

    if not recentre:
        card.paste(src, ((size[0] - src.size[0]) // 2, (size[1] - src.size[1]) // 2))

        return card

    lit = np.array(src).astype(np.float32).sum(axis=2) > threshold
    cx = int((np.arange(src.size[0])[None, :] * lit).sum() / lit.sum())
    cy = int((np.arange(src.size[1])[:, None] * lit).sum() / lit.sum())
    card.paste(src, (size[0] // 2 - cx, size[1] // 2 - cy))

    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', required=True, help='dark render from gallery_gif.py')
    parser.add_argument('--logo', default=DEFAULT_LOGO)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--logo-size', type=int, default=560)
    parser.add_argument('--feather', type=int, default=160)
    parser.add_argument('--recentre', action='store_true',
                        help='shift a loosely framed render onto its lit pixels')
    # the showcase camera leaves the lit mass on the left, so the logo goes right of centre
    # and its own streamlines fill what would otherwise be an empty half
    parser.add_argument('--logo-dx', type=int, default=0)
    args = parser.parse_args()

    card = background(args.frame, SIZE, args.recentre)
    logo = Image.open(os.path.normpath(args.logo)).convert('RGB')
    logo = logo.resize((args.logo_size, args.logo_size), Image.LANCZOS)

    card.paste(logo,
               ((SIZE[0] - args.logo_size) // 2 + args.logo_dx,
                (SIZE[1] - args.logo_size) // 2),
               feather_mask(args.logo_size, args.feather))

    out = os.path.normpath(args.out)
    card.save(out, optimize=True)

    print('%s  %dx%d  %.0f KB' % (out, SIZE[0], SIZE[1], os.path.getsize(out) / 1e3))


if __name__ == '__main__':
    main()
