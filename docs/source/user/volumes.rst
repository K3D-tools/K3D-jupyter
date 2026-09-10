.. _volumes:

=======
Volumes
=======

A ``volume`` is the object whose appearance is decided almost entirely by how light is
transported rather than by what a surface does, which makes it the one place where the
choice of renderer changes the picture rather than the polish. Everything on this page
applies to ``k3d.volume``; ``mip`` follows it where it says so.

Materials
---------

Volumetric objects (``volume``, ``mip``) carry the same two knobs for the
specular highlight of their isodensity surface (default ``roughness=0.25``) -
lower roughness makes noisy gradients sparkle like wet tissue, which may even
be desired. ``metalness`` tints and strengthens the highlight with the
transfer-function colour; it never darkens the body, because a volume has no
environment reflection to replace the lost diffuse light with.

Light and the environment
-------------------------

Volumetric data (``volume``, ``mip``) and the ``points`` 3d impostors read the same
environment: diffuse light from the map's spherical harmonics plus one dominant
directional light distilled from it, so a directional HDRI models volumes consistently
with every mesh in the scene.

In ``advanced`` a volume also contributes to the occlusion pass - the shell where its
accumulated opacity crosses one half - so dense structures cast and receive contact
shadows like real surfaces. The two knobs are ``plot.ao_radius`` and
``plot.ao_strength``, described in :ref:`renderers`.

Composing with geometry
-----------------------

Since 3.0.0 a ``volume`` composes correctly with meshes that intersect it
when depth peeling is enabled (``plot.depth_peels >= 3`` - fewer layers make
the segmentation too coarse to be predictable). The ray march is split into
segments bounded by the peel layers, so geometry inside the volume occludes
and is occluded sample-accurately, in both renderers:

.. k3d_plot ::
  :filename: plots/renderers_volume_peel_plot.py

Volumetric data (``volume``, ``mip``) and the ``points`` 3d impostors read the
same environment: diffuse light from the map's spherical harmonics plus one
dominant directional light distilled from it, so a directional HDRI models
volumes consistently with every mesh in the scene.

A volume through each renderer
------------------------------

A volume is where the three differ most, because it is the one object whose appearance is
almost entirely a question of how light is transported rather than of what the surface
does. The same cardiac CT below, at the same ``alpha_coef``, the same transfer function,
the same camera and the same environment:

.. Sphinx copies an image into _images and rewrites its src, but leaves :target: verbatim -
   it is a URI, not an image reference. Hence the explicit path, which is relative to this
   page's depth: a page one level under html/.

.. list-table::
   :widths: 50 50

   * - .. image:: renderers_heart_simple.png
          :width: 100%
          :target: ../_images/renderers_heart_simple.png

     - ``simple``. Four lights that follow the camera and a ray march that shades every
       sample where it stands. Nothing inside the medium casts onto anything else, so
       tissue at the back of the chest is as bright as tissue at the front and the image
       carries no order of depth: the chambers, the vessels behind them and the far ribs
       all sit on one plane of brightness.

   * - .. image:: renderers_heart_simple_shadow.png
          :width: 100%
          :target: ../_images/renderers_heart_simple_shadow.png

     - ``simple`` with ``shadow='on-demand'`` on the volume - the only change. A light map
       is built once and the march reads it, so dense tissue darkens what lies behind it.
       Look at the middle of the frame, where the heart's body meets what is behind it:
       the far tissue drops back and the near vessels separate from it. The measured change
       is 42% of pixels, over a hundred levels in the deepest recesses, and almost nothing
       at the edges - one light map, not a simulation.

   * - .. image:: renderers_heart_advanced.png
          :width: 100%
          :target: ../_images/renderers_heart_advanced.png

     - ``advanced`` at ``ao_radius=0.1``, ``ao_strength=0.5``. All the light comes from the
       environment map, and a GTAO pass grounds what the volume contributes to it - the
       shell where accumulated opacity crosses one half. The crevices between vessels and
       the undersides of the ribs deepen, the chambers stop reading as blown out, and the
       whole frame gains contrast without anything moving.

   * - .. image:: renderers_heart_cinematic.png
          :width: 100%
          :target: ../_images/renderers_heart_cinematic.png

     - ``cinematic`` at 128 samples with ``cinematic_denoise=2.0``. Delta tracked as a
       participating medium: light that enters the tissue scatters inside it and is
       attenuated on the way out, so interiors go dark against the vessels that catch
       the light, and a density gradient steep enough to be a boundary shades as a
       surface rather than as gas. The ribs are defocused because the aperture is a real
       one. What grain is left is the price of a budget chosen for a docs build rather
       than for a final render.

Every one of the four is a full HD render - click it to open it at full size, because the
difference between ``advanced`` and ``cinematic`` lives in detail a page-width figure loses.

Two things are deliberately not held constant, and both are worth knowing. The volume's
``shadow`` is off in the first image and on in the second, which is the point of having
both. And ``light_scale`` is 2.25, which lifts only the traced image, because the raster
shader does not read it: the march lights every sample locally and counts the environment
twice while the tracer attenuates, so at this density an unlifted traced image is much the
darkest of the four and the comparison would be about exposure instead of about light.

.. k3d_plot ::
   :filename: plots/renderers_heart_simple.py
   :screenshot:

.. k3d_plot ::
   :filename: plots/renderers_heart_simple_shadow.py
   :screenshot:

.. k3d_plot ::
   :filename: plots/renderers_heart_advanced.py
   :screenshot:

.. k3d_plot ::
   :filename: plots/renderers_heart_cinematic.py
   :screenshot:

In the path tracer
------------------

A ``volume`` is part of the light simulation. Its bounding box sits in the
acceleration structure as the boundary of a medium, and inside it every ray -
camera rays, bounced rays, shadow rays - is tracked through the 3D texture with
the same transfer function the other renderers use. How much matter a ray meets
matches the raster exactly: ``alpha_coef`` and the opacity function give the same
optical depth here as in ``advanced``, so a volume tuned there keeps its density
here and only the lighting changes. Geometry inside the gas is shadowed by it and
occludes it, the gas appears in reflections, and light scattered inside it is
the environment's.

Where the density changes sharply the medium is shaded as a surface: at every
collision the tracer measures the change of the normalised intensity over one
``gradient_step`` and, with probability :math:`1 - e^{-8\,m}` for a change
:math:`m`, treats the point as a rough dielectric facing the falling density,
with the transfer function colour and the volume's own ``roughness`` and
``metalness``. Bone and skin in a CT get highlights, Fresnel and an orientation;
soft tissue with gentle gradients stays gas. Gas events scatter forward with a
Henyey-Greenstein phase function (asymmetry 0.85, as measured tissue does), so light
reaches deeper than an isotropic phase would let it:

.. k3d_plot ::
  :filename: plots/renderers_volume_cinematic_plot.py

What to expect from a physically traced volume, as opposed to the ray march:

* the march shades every sample locally and without occlusion, and lights it with
  the environment counted twice - its harmonics plus a directional light distilled
  from the same map - so a dense volume glows with its interior colours. The tracer
  attenuates instead, so the two agree only where the volume is thin: on a CT scan
  at ``alpha_coef`` 15 the traced image is the brighter of the two, at 200 it is a
  third of the march. Lower ``alpha_coef`` or a brighter colour map does here what
  it does to real fog, and ``roughness`` and ``metalness`` shape the highlights;
* the medium has its own exposure, ``light_scale`` on the volume, because a
  brighter environment lifts the whole scene while a dense volume needs more
  light than the geometry around it. It multiplies what reaches an event inside
  the medium, so every ratio in the image survives - a crevice stays as much
  darker than an exposed surface as it was - and only the exposure moves. It
  cannot rescue a pitch-black deep interior: where no light arrives there is
  nothing to multiply, and ``cinematic_bounces`` is what carries light in there;
* ``samples`` is not used: there is no fixed step to set. The tracker draws the
  distance to the next collision from the density itself, which needs no sampling
  rate and introduces no error of its own - ``cinematic_samples`` trades noise for
  time instead, and ``alpha_coef`` sets the cost, since it sets how often a ray is
  interrupted;
* the surface decision is made per collision from the local gradient, so a
  transfer function that ramps gently over a boundary gives a softer, gassier
  edge than one that steps; ``gradient_step`` sets the scale of the gradient, in
  the volume's mean edge length, and is the same knob the other renderers use;
* cost follows the density in front of the ray, not the densest voxel in the box.
  The tracker keeps a grid of macrocells, each holding the largest extinction any
  point inside it can have, and walks that grid: a cell of air is crossed in one
  step, and a thin region is stepped at its own rate. On a 512x512x1319 CT scan
  with an opacity function rising across its colour range, 40% of the cells hold
  nothing at all and the mean majorant is a ninth of the global one, so an average
  ray tests about a ninth as many collisions - in exchange for a few hundred cell
  crossings, which are far cheaper than a sample of a volume too large to sit in
  cache. A ray straight through the densest bone still saves close to half, which is
  the honest limit: dense matter costs, and ``alpha_coef`` still sets how often a ray
  inside it is interrupted;
* one volume per plot is traced; any further ``volume`` stays on the raster
  overlay with a warning. So does a volume with a mask (``mask``,
  ``mask_opacities``): the medium does not read the mask, and tracing the volume as
  if the mask were not there would be worse than not tracing it;
* the medium needs two more texture units than the tracer's surfaces do, one for
  its data and one for its majorant grid; a context that cannot provide fifteen
  keeps the volume on the raster overlay and says so once in the console.

``mip`` is a maximum-intensity projection, a diagnostic view rather than a
physical one, and stays what it was: ray-marched as in ``advanced``, stopping at
the first traced surface, and composited over the traced image outside the
light simulation.
