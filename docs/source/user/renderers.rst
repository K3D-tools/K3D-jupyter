.. _renderers:

Renderers
=========

Since 2.19.0 every plot carries a ``renderer`` switch:

.. code-block:: python3

    plot = k3d.plot(renderer='advanced')
    # or at any moment:
    plot.renderer = 'advanced'

``simple`` (default)
    The classic rasteriser: a fixed rig of four lights that follows the camera.
    Fast, stable, and unchanged - every existing notebook renders exactly as before.

``advanced``
    Image-based lighting plus ambient occlusion. All light comes from an
    environment map, materials are physically based, and a GTAO pass grounds the
    geometry with contact shadows. Switching back and forth is a single
    assignment and never rebuilds the scene.

The golden rule: **the renderer changes the light, never what you asked to
draw**. Unlit primitives - ``points`` with ``dot``/``flat`` shaders,
``line(simple)``, ``line(thick)``, labels, texts, wireframes - are a deliberate
choice of no shading and look identical in both renderers.

Materials
---------

All lit objects use physically based materials with two knobs:

``roughness``
    0.0 = polished mirror-like highlight, 1.0 = fully matte. Default 0.4.

``metalness``
    0.0 = dielectric, 1.0 = metal (the surface reflects only its environment,
    tinted by its own colour). Default 0.0.

``shininess`` was removed in 2.19.0. The equivalent conversion is
``roughness = sqrt(2 / (shininess + 2))``; passing ``shininess`` raises a loud
``TraitError`` with that formula, and legacy ``.k3d`` snapshots are converted
automatically on load.

Environments
------------

In ``advanced`` the environment map is the only light source. Every map is
energy-normalised, so the environment carries the *shape* of the light while
``plot.lighting`` stays the exposure knob.

.. code-block:: python3

    plot.environment = 'studio'            # procedural preset
    plot.environment = 'burnt_warehouse'   # photographic catalog (Poly Haven, CC0)
    plot.environment = my_hdr_array        # any (H, W, 3) float32 equirect
    plot.show_environment = True           # show the map as the background
    plot.environment_rotation = np.pi / 3  # spin it around the scene's up axis

Procedural presets (``neutral`` - the default, ``studio``, ``outdoor``) travel
as plain names and are generated deterministically on the CPU. The photographic
catalog ships with the package:

.. code-block:: python3

    import k3d.environments
    k3d.environments.available()
    # ['autoshop_01', 'brown_photostudio_02', 'burnt_warehouse',
    #  'moonless_golf', 'noon_grass']

Volumetric data (``volume``, ``mip``) and the ``points`` 3d impostors read the
same environment: diffuse light from the map's spherical harmonics plus one
dominant directional light distilled from it, so a directional HDRI models
volumes consistently with every mesh in the scene.

Ambient occlusion
-----------------

``advanced`` always includes a GTAO pass with spatial denoising. It has no
parameters to tune: the occlusion radius scales with the scene's bounding box,
the result is deterministic, and screenshots are seam-free at any
``rendering_steps``. Real surfaces occlude; volumes and MIPs contribute the
shell where their accumulated opacity crosses one half, so dense structures
cast and receive contact shadows too.

Tone mapping
------------

.. code-block:: python3

    plot.tone_mapping = 'agx'   # or 'aces'; 'none' is the default

Filmic curves compress bright HDR highlights - useful with high-contrast
photographic environments.
