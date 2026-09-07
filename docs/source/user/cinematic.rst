.. _cinematic:

Cinematic rendering
===================

.. warning::
    **Experimental.** ``cinematic`` is new in 3.0.0 and not yet on the same
    footing as the other two renderers: the trait names and their defaults may
    change, the image a given scene produces may change between versions, and
    the coverage gaps listed below are real rather than temporary oversights
    (``volume_slice`` is not drawn, ``mip`` stays outside the light simulation).
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
    plot.cinematic_seed = None           # None: fresh noise each time; an int: repeatable
    plot.cinematic_denoise = 0.0         # filter strength in noise sigmas, 0 is off
    plot.cinematic_bokeh_size = 0.0      # aperture diameter in scene units, 0 is a pinhole
    plot.cinematic_focus_distance = 0.0  # distance from the camera; 0 is as far as its target
    plot.cinematic_aperture_blades = 0   # 0 is a round iris, 3 to 16 a polygonal one

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

``cinematic_bokeh_size``, ``cinematic_focus_distance``, ``cinematic_aperture_blades``
    The lens, described below. 0 is a pinhole and leaves the image identical to the
    other renderers.

``cinematic_seed``
    ``None``, the default, draws fresh noise for every accumulation, so two renders of
    the same scene differ in their residual grain the way two photographs do. An
    integer pins the noise: the same plot then renders the same image, sample for
    sample, across page loads and machines - what a reference-image test suite or a
    frame-by-frame animation needs, and what a notebook does not. Different seeds
    give different, equally valid patterns.

.. note::
    Path tracing produces high dynamic range: bounced light between bright
    surfaces genuinely exceeds 1.0, and without a tone curve those values clip.
    A yellow menger sponge - all cavities, all bounce - blows out about 7% of its
    pixels at ``tone_mapping='none'`` and none at all with ``'aces'``. If a
    cinematic render looks hot where ``advanced`` looked fine, reach for
    ``plot.tone_mapping`` before ``plot.lighting``.

Anti-aliasing
~~~~~~~~~~~~~

``plot.antialias`` is a rasteriser's tool and ``cinematic`` does not use it. In the
other two renderers a screenshot is drawn up to 32 times with the camera nudged by a
fraction of a pixel and the results averaged, because a rasteriser has no other way to
get an edge sample. The tracer never enters that loop: it already sends each ray
through a random point inside the pixel, weighted by a tent filter rather than a box,
so every sample you pay for is also an anti-aliasing sample. A 64-sample render has
reconstructed its edges from 64 subpixel positions - more than the jitter table offers
at any setting, and for no extra time.

Leave it at 0. A non-zero value still asks the browser for a multisampled framebuffer,
and cinematic ends on a fullscreen quad with no geometry edge to multisample: the image
comes out the same and the buffer is not free, which at 4K is worth not paying for. The
one thing it still reaches is the rasterised axes helper composited into the corner, so
it matters only while ``plot.axes_helper`` is on.

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

Rendering an animation
~~~~~~~~~~~~~~~~~~~~~~

Every frame is an independent accumulation - nothing is reused between them - so a
sequence has a failure mode a single image does not: the brightness of the whole frame
walks about slightly from one to the next, and bone and tissue appear to flicker even
though the scene is changing smoothly.

The cause is the sampler, not the scene. All pixels share one stratified sequence, each
offset by its own blue-noise value, so the residual error of a frame is **coherent
across it** rather than independent per pixel - and with ``cinematic_seed`` left unset
that sequence is reshuffled from ``Math.random`` for every frame. Each frame therefore
carries its own small, whole-frame bias.

.. code-block:: python3

    plot.cinematic_seed = 1      # the same sequence for every frame
    plot.cinematic_denoise = 2.0

Pinning the seed removes the flicker: every frame draws the identical sequence, so the
coherent part is identical too and only what you actually animate changes. On its own
it trades one artefact for another - the grain is seeded from the pixel's own
coordinate, so it stops moving and sits still on the screen while the image travels
underneath it. The denoiser is what makes the pair work, because it removes the grain
that pinning has just made stationary.

Do not vary the seed with the frame number to avoid the stuck-grain look. That puts the
flicker straight back: a different sequence per frame is exactly what caused it.

One more thing a frame loop needs. It is not specific to ``cinematic`` - it is just that
a path traced frame takes long enough to make the race easy to hit:

.. code-block:: python3

    for i in range(frames):
        set_up_frame(i)                            # camera, colour range, whatever moves
        headless.sync(hold_until_refreshed=True)   # not sync() - see below

        with open('frame_%06d.png' % i, 'wb') as f:
            f.write(headless.get_screenshot(True))

``sync()`` on its own returns as soon as it has *asked* the page to refresh. The state
travels to the browser over an asynchronous request, so a screenshot taken immediately
afterwards renders whichever scene the page happens to be holding - sometimes the one
you just set, sometimes the previous one. The symptom is two byte-identical files in
the middle of a sequence. ``hold_until_refreshed=True`` waits for the page to confirm it
has the new state, which is what every reference image in this library uses. The rest of
that API - the drivers, the resolution, the diagnostics - is on its own page,
:ref:`headless`.

Denoising
~~~~~~~~~

Monte Carlo error falls as one over the square root of the sample count, so every
halving of the grain costs four times the samples. Past a few hundred a render keeps
improving and stops looking like it: the change between one budget and the next drops
below what an eye can see long before the noise is gone.

.. code-block:: python3

    plot.cinematic_denoise = 2.0   # 0 is off

``cinematic_denoise`` is measured in standard deviations of the noise the renderer
estimates for each pixel, and 0 is off - the only value that leaves the image exactly
as it was traced. Around 2 removes most of the grain a moderate budget leaves behind.
By about 4, bone in a CT scan starts to look waxy: the grain and the trabecular texture
underneath it are the same size, and they leave together.

What makes this different from a general-purpose blur is what guides it. The renderer
splits its own accumulation in two by sample parity and reads the spread between the
halves, which gives the variance of every pixel for free - the samples were traced
anyway. Where a pixel has settled the filter passes it through untouched; where it has
not, it averages. The usual alternative, edge-stopping on depth and normals, does not
work inside a volume at all: there is no first surface, the first collision is a random
variable, and at low sample counts those buffers are themselves noise. Accumulated
opacity is used alongside the variance, because without it the filter cannot tell a
dark part of the medium from the black behind it.

Three consequences worth knowing.

It is worth roughly four times the samples on a volume and nothing at all on a
converged render - it removes error, and a converged image has none left to remove.
The switch has to be on **before** the render: the halves it is guided by fill one
sample at a time, so turning it on afterwards finds them empty and restarts the
accumulation. And it costs memory - three float buffers at the resolution being
filtered, which at a 4K screenshot is not free.

.. k3d_plot ::
   :filename: plots/cinematic_denoise_plot.py
   :screenshot:

Depth of field
~~~~~~~~~~~~~~

The other two renderers see through a pinhole: everything is in focus because every
ray starts at a single point. ``cinematic`` traces a real aperture, so it can put a
plane in focus and let the rest fall away - which is a property of the lens, not of
any object in the scene, and lives on the plot for that reason.

.. code-block:: python3

    plot.cinematic_bokeh_size = 0.4    # scene units - a CT scan 400 units wide needs a bigger one
    plot.cinematic_focus_distance = 0  # 0: whatever the camera is pointed at

``cinematic_bokeh_size`` is the diameter of the aperture **in the units of your own
scene**, the same units as ``bounds`` or ``point_size``, because a plot has no idea
what a millimetre is. It is 0 by default, and that is the only value that leaves the
image exactly as the other renderers draw it - opening it beyond zero costs one shader
recompile, then nothing.

``cinematic_focus_distance`` is how far in front of the camera the sharp plane sits.
Left at 0 it follows the camera's own target, so the plot stays sharp where you are
looking and the focus tracks the orbit; set it explicitly to focus in front of or
behind that.

``cinematic_aperture_blades`` shapes the iris. 0, the default, is a perfect circle;
3 to 16 give an aperture of that many sides, which is what makes an out-of-focus
highlight read as a hexagon rather than a disc. It does nothing while the aperture is
closed.

Two things follow from this being a real aperture rather than a blur applied
afterwards. It costs no extra samples - the aperture is sampled along with everything
else - but it *does* need more of them, because a defocused region is an average over
a wider set of paths and converges more slowly. And it cannot be undone in
post: what the lens did not resolve was never traced.

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
   * - ``volume``
     - Traced as a participating medium: the box enters the acceleration
       structure as the medium boundary and rays are tracked through the 3D
       texture inside it. One volume per plot; further ones fall back to the
       raster overlay with a warning. See below.
   * - ``mip``
     - Ray-marched as in ``advanced`` and composited over the traced image.
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

Volumes are the biggest difference between this renderer and the other two, and they
have their own page: :ref:`volumes`.

Requirements and failure
------------------------

``cinematic`` needs WebGL2 with renderable float textures. When the browser
cannot provide them, switching to it **fails loudly**: an error overlay names
the reason and the ``renderer`` trait reverts to its previous value. There is no
silent fallback to another renderer - a plot that says ``cinematic`` is always
path traced.

Cost scales with resolution, sample budget and bounce count. On a software
renderer (CI, remote sessions without a GPU) a converged frame takes seconds to
minutes; the library's own reference images use 16 samples at half
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
