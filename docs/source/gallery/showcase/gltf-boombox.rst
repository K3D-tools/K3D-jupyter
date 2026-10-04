glTF boombox
============

.. admonition:: References

    - :ref:`glb`
    - :ref:`mesh`
    - :ref:`cinematic`

A model made for real-time renderers, read straight from its ``.glb``. Everything it carries is
an ordinary K3D mesh parameter after reading: the colour texture, the normal map that gives the
speaker grille its holes, the metalness-roughness map that makes the trim chrome and the body
plastic, the occlusion map darkening the creases, and the emissive map that lights the PLAY
button. Nothing is left out, so :func:`k3d.glb` reads it without a warning.

The model is two centimetres wide, so ``scaling`` makes it a hundred times larger - the
transform arguments of every K3D factory place the whole model. The textures are halved here
only because this page carries every one of them inline; read from the file they are used at
full size.

The thumbnail is the same scene under :ref:`cinematic`: the PLAY button then lights the floor
in front of it, and the chrome reflects the room.

.. code-block:: python3

    import io

    import numpy as np
    from PIL import Image

    import k3d
    from k3d.helpers import download

    # Model: BoomBox, Microsoft, CC0
    filename = download('https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/'
                        'main/Models/BoomBox/glTF-Binary/BoomBox.glb')

    boombox = k3d.glb(filename, scaling=[100, 100, 100])


    def halved(data):
        image = Image.open(io.BytesIO(data))
        image = image.resize((image.width // 2, image.height // 2), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()


    for part in boombox:
        for name in ('texture', 'emissive_map', 'normal_map', 'metalness_roughness_map',
                     'occlusion_map'):
            if getattr(part, name):
                setattr(part, name, halved(getattr(part, name)))

    bounds = np.array([part.get_bounding_box() for part in boombox])
    low, high = bounds[:, 0::2].min(axis=0), bounds[:, 1::2].max(axis=0)
    centre = (low + high) / 2
    size = float((high - low).max())

    plot = k3d.plot(renderer='advanced',     # or 'cinematic'
                    environment='studio',
                    tone_mapping='aces',
                    grid_visible=False,
                    camera_auto_fit=False,
                    background_color=0x1E2126,
                    cinematic_samples=256,
                    cinematic_bounces=6)
    plot += boombox

    span = 1.5 * size
    floor_z = float(low[2])
    plot += k3d.mesh(np.array([[centre[0] - span, centre[1] - span, floor_z],
                               [centre[0] + span, centre[1] - span, floor_z],
                               [centre[0] + span, centre[1] + span, floor_z],
                               [centre[0] - span, centre[1] + span, floor_z]], np.float32),
                     np.array([[0, 1, 2], [0, 2, 3]], np.uint32),
                     color=0x3A3F47, roughness=0.6, name='floor')

    eye = centre + np.array([0.55, -1.2, 0.45]) * size
    plot.camera = [*eye, *centre, 0, 0, 1]

    plot.display()

The parts of a larger model - every primitive of every node - each get their own folder in the
panel, grouped under the file's name; ``examples/gltf_import.ipynb`` walks through one.

.. k3d_plot ::
  :filename: plots/gltf_boombox_plot.py
