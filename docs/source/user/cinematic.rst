.. _cinematic:

Cinematic rendering
===================

.. warning::
    **Experimental.** ``cinematic`` is new in 3.0.0 and not yet on the same
    footing as the other two renderers: the trait names and their defaults may
    change, the image a given scene produces may change between versions, and
    the coverage gaps listed below are real rather than temporary oversights
    (``volume_slice`` is not drawn, volumes stay outside the light simulation).
    ``simple`` and ``advanced`` remain the stable choices; please report what
    breaks.

.. code-block:: python3

    plot = k3d.plot(renderer='cinematic')

Where :ref:`advanced <renderers>` approximates indirect light with an occlusion
pass, ``cinematic`` traces it: rays scatter off surfaces up to
``cinematic_bounces`` times, gathering colour from the environment and from each
other. Soft shadows, mirror and glossy reflections, and colour bleeding between
nearby objects all appear without a single extra knob - they are consequences of
the simulation rather than effects layered on top of it.

The image is progressive: one sample per animation frame, with a counter in the
corner, until it reaches ``cinematic_samples`` - a hard ceiling, after which the
loop stops and an idle plot costs nothing. Any change to the camera, the scene or
the lighting abandons the accumulation and starts it again from sample zero, so
what you see always describes the current state. While you drag the camera the
frame is rasterised instead (the same picture ``advanced`` would draw, minus the
occlusion pass), so the view follows the mouse; path tracing resumes the moment
the camera settles. Screenshots always render the full budget, so an exported
image is as clean as the budget allows regardless of what the interactive view
had reached.

The parameters
--------------

.. code-block:: python3

    plot.cinematic_samples = 64          # accumulation budget, [1, 100000]
    plot.cinematic_bounces = 6           # light bounces, [1, 32]
    plot.cinematic_glossy_filter = 0.25  # widen glossy lobes after a rough bounce, [0, 1]

``cinematic_samples``
    How many samples the accumulation gathers before it parks. Noise falls off as
    the square root of this number, so 4x the samples means half the noise: the
    step from 32 to 128 is plainly visible, the one from 512 to 2048 rarely is.
    Cost is linear in it. The ceiling is deliberately far above anything
    interactive, because a final render is worth waiting for - and because the
    loop stops there rather than burning a GPU forever.

``cinematic_bounces``
    How far light is followed. 1 is direct lighting only: no colour bleeding, no
    reflection of one object in another, and interiors go black. 6 is enough for
    ordinary scenes; a closed white room or a stack of glossy surfaces keeps
    getting brighter up to 12 or so. Cost grows with it, though sub-linearly -
    paths that leave the scene stop early.

``cinematic_glossy_filter``
    Firefly control, described below. 0 leaves light transport unbiased.

.. note::
    Path tracing produces high dynamic range: bounced light between bright
    surfaces genuinely exceeds 1.0, and without a tone curve those values clip.
    A yellow menger sponge - all cavities, all bounce - blows out about 7% of its
    pixels at ``tone_mapping='none'`` and none at all with ``'aces'``. If a
    cinematic render looks hot where ``advanced`` looked fine, reach for
    ``plot.tone_mapping`` before ``plot.lighting``.

Fireflies
~~~~~~~~~

A polished surface lit by a small very bright source - metal under a sunny HDRI,
typically - throws **fireflies**: isolated bright pixels left by the rare path
that happens to reach the sun through a mirror. They fade as the square root of
the sample count, which is to say hardly at all.

``cinematic_glossy_filter`` widens a glossy lobe in proportion to the roughness
already gathered along the path. A specular seen directly is unaffected - nothing
has accumulated yet - while the path that hits a rough surface first and a mirror
second gets spread out, and the speckle with it. That is why it defaults to 0.25:
the bias is invisible where you look straight at a reflection, and it removes the
artefact where the artefact lives.

Its limit follows from the same rule. A chain of *smooth* surfaces accumulates
almost no roughness, so a mirror floor reflecting a polished model keeps its
fireflies at any setting - raising the filter does nothing there. What helps is
giving one of the two surfaces some roughness, or choosing an environment whose
brightest spot is less concentrated than a sun.

Environments are the light
--------------------------

There are no light objects in ``cinematic``. The environment map is the only
source of illumination, and it is what every reflective surface reflects, so
choosing it is the single biggest decision about how a plot looks - more than any
material parameter.

It is not the backdrop, though: behind the scene you get ``plot.background_color``,
exactly as in the other two renderers. The environment lights the model and shows
up in its reflections; the space behind the data stays yours. A photograph of a
warehouse behind a plot would look striking and say nothing - the light it casts is
what changes how a surface reads, and that is the part worth having.

.. code-block:: python3

    plot.environment = 'studio'            # procedural preset
    plot.environment = 'venice_sunset'     # photographic catalog (Poly Haven, CC0)
    plot.environment = my_hdr_array        # any (H, W, 3) float32 equirect
    plot.environment_rotation = np.pi / 3  # spin it around the scene's up axis
    plot.lighting = 1.5                    # exposure, not a light count
    plot.tone_mapping = 'aces'             # filmic curve for the highlights

Every map is energy-normalised, so the environment carries the *shape* of the
light while ``plot.lighting`` stays the exposure knob. Rotating it moves the
highlights without changing their intensity, which is often the quickest way to
make a specific surface read well.

The same gold dragon on a polished floor, under six environments, at 256 samples
each. Nothing changes between these images except ``plot.environment`` - so every
difference you see is the light itself and what the metal reflects of it:

.. list-table::
   :widths: 50 50

   * - .. image:: cinematic_env_neutral.png
          :width: 100%

       ``neutral`` - the default, and procedural. The light has no story, which is
       exactly what you want when judging a material rather than a mood.
     - .. image:: cinematic_env_autoshop_01.png
          :width: 100%

       ``autoshop_01`` - rows of ceiling strips. Each one draws a long highlight
       down the spine, and the hall itself appears in the flanks.
   * - .. image:: cinematic_env_brown_photostudio_02.png
          :width: 100%

       ``brown_photostudio_02`` - one big window against a dark room: a single
       soft key light, deep falloff, and very little fill.
     - .. image:: cinematic_env_burnt_warehouse.png
          :width: 100%

       ``burnt_warehouse`` - warm brick and small openings. Contrasty and dim;
       gold reads almost brown where nothing reaches it.
   * - .. image:: cinematic_env_moonless_golf.png
          :width: 100%

       ``moonless_golf`` - a whole sky at dusk acting as one enormous softbox,
       lighting the model evenly from above.
     - .. image:: cinematic_env_venice_sunset.png
          :width: 100%

       ``venice_sunset`` - a low sun. The strongest directional highlight of the
       six and the most saturated colour cast.

.. k3d_plot ::
   :filename: plots/cinematic_env_neutral.py
   :screenshot:

.. k3d_plot ::
   :filename: plots/cinematic_env_autoshop_01.py
   :screenshot:

.. k3d_plot ::
   :filename: plots/cinematic_env_brown_photostudio_02.py
   :screenshot:

.. k3d_plot ::
   :filename: plots/cinematic_env_burnt_warehouse.py
   :screenshot:

.. k3d_plot ::
   :filename: plots/cinematic_env_moonless_golf.py
   :screenshot:

.. k3d_plot ::
   :filename: plots/cinematic_env_venice_sunset.py
   :screenshot:

.. code-block:: python3

    import k3d.environments
    k3d.environments.available()
    # ['autoshop_01', 'brown_photostudio_02', 'burnt_warehouse',
    #  'moonless_golf', 'venice_sunset']

.. note::
    The photographic maps live in the Python package, so a kernel-less page
    cannot resolve their names. An exported HTML snapshot therefore offers only
    what it can regenerate: the procedural presets plus the map that was baked
    into it at export time. A page may widen that list by including the sideload
    script generated by ``k3d.environments.save_js(path)`` next to
    ``standalone.js``.

Ambient-occlusion knobs are absent from the panel here: occlusion is not
approximated, it is traced.

What changes shape-for-shape
----------------------------

A path tracer needs surfaces with area, so objects drawn as screen-space
impostors are rebuilt as real geometry. The result keeps the shape you asked
for; the differences worth knowing:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - In ``cinematic``
   * - ``mesh``, ``stl``, ``surface``, ``marching_cubes``, ``voxels``,
       ``texture``
     - Traced as they are.
   * - ``points`` (any shader)
     - Merged spheres of real geometry. Sphere detail adapts to the point count
       and is capped by a triangle budget, so very large clouds render coarser
       spheres. ``dot`` has no world-space size at all (it is a pixel count), so
       ``point_size`` is taken as its diameter. Per-point opacity is ignored.
   * - ``line``, ``lines`` (``simple``/``thick``)
     - Tubes of world-space width. ``thick`` extrudes its full width on screen,
       so its tube radius is ``width / 2``, while the ``mesh`` shader already
       treats ``width`` as a radius - a ``thick`` line and a ``mesh`` line of
       the same ``width`` differ by 2x, exactly as they do when rasterised.
   * - ``vectors``, ``vector_field``
     - Shafts become tubes of radius ``line_width / 2``, heads stay cones.
   * - ``texture_text``
     - Camera-facing quads, frozen in the orientation they had when the
       accumulation started; they do not turn with the camera mid-frame.
   * - ``text``, ``text2d``, ``label``
     - Unchanged: HTML overlays drawn on top of the finished frame.
   * - ``volume_slice``
     - Not rendered (a warning says so). A slice paints its cut plane with its
       own shader and carries no depth-segment mechanism, so it can neither be
       traced nor composited correctly; use ``simple`` or ``advanced`` for slice
       views.
   * - Unlit primitives
     - Lit. A path tracer has no unlit surface, so ``dot``/``flat`` points and
       simple lines pick up shading they never had in the other renderers.
   * - The grid
     - Not drawn.

Volumes and MIPs
----------------

The path tracer knows only homogeneous fog, so ``volume`` and ``mip`` keep the
ray march they use in ``advanced`` - lit by the same environment harmonics -
and composite over the traced image. The march stops at the first traced
surface, so geometry inside or behind a volume occludes correctly and gas in
front of geometry dims it:

.. k3d_plot ::
  :filename: plots/renderers_volume_cinematic_plot.py

The limits of this hybrid, in exchange for keeping volumes at all:

* a volume does not appear in reflections or refractions, and casts no light or
  shadow onto geometry - global illumination does not see the gas;
* geometry seen *through* a reflection is not dimmed by gas in front of it,
  although geometry seen directly is;
* ``mip`` is a maximum-intensity projection, a diagnostic view rather than a
  physical one; in ``cinematic`` it stays exactly that, composited outside the
  light simulation.

Requirements and failure
------------------------

``cinematic`` needs WebGL2 with renderable float textures. When the browser
cannot provide them, switching to it **fails loudly**: an error overlay names
the reason and the ``renderer`` trait reverts to its previous value. There is no
silent fallback to another renderer - a plot that says ``cinematic`` is always
path traced.

Cost scales with resolution, sample budget and bounce count. On a software
renderer (CI, remote sessions without a GPU) a converged frame takes seconds to
minutes; the library's own reference images use 32 samples at a quarter
resolution for exactly that reason.

Before the first sample the scene needs a ray-tracing acceleration structure,
rebuilt whenever the geometry changes. Past a hundred thousand triangles that
build moves to a worker - the counter reports its progress and the camera keeps
responding on rasterised frames meanwhile. In a notebook the worker script comes
from the kernel and nowhere else, which keeps an air-gapped deployment
air-gapped; a standalone page looks for it next to the bundle it loaded, then on
unpkg for that same version. When none of those answers - an unpublished build,
a network without a route out - the structure is built on the main thread and the
page stops responding for as long as that takes.
