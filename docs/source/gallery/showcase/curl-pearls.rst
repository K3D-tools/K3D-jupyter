Curl-noise pearls
=================

.. admonition:: References

    - :ref:`plot`
    - :ref:`points`

Strings of pearls advected through a divergence-free curl-noise field. The
vector potential is a handful of long-wavelength sinusoids (laminar sheets)
plus medium waves (turbulence), enveloped radially so the streamlines curve
along a sphere instead of escaping it. Ribbons come from seeding lines of
points across the flow; beads are deposited densely along each streamline
and taper towards the tail.

The beads are ``points`` with ``shader='3d'`` - analytic sphere impostors,
which cast and receive the advanced renderer's ambient occlusion. The blocks
behind them are there for the same pass: a plane offers only a contact shadow,
while the concave corners where blocks meet each other are where occlusion
reads strongest. One radius serves both scales (``ao_radius=0.05``,
``ao_strength=2.4``): the crevices between strands and the corners between
blocks. Everything here is a few seconds of numpy with no data files involved,
and the flow field itself is worked through in ``examples/curl_noise_pearls.ipynb``.

.. code-block:: python3

    plot = k3d.plot(grid_visible=False, camera_auto_fit=False,
                    background_color=0x08090B,
                    renderer='advanced', environment='neutral',
                    lighting=2.1, ao_radius=0.05, ao_strength=2.4)

    plot += k3d.points(positions, point_sizes=point_sizes, shader='3d',
                       colors=point_colors, roughness=0.35)
    plot += k3d.mesh(block_vertices, block_indices,
                     color=0x35506B, roughness=0.55)
    plot.camera = [1.75, -1.75, 1.1, 0, 0, 0, 0, 0, 1]
    plot.display()

.. k3d_plot ::
  :filename: plots/curl_pearls_plot.py
