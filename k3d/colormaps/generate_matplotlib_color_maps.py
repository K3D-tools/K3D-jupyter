import numpy as np
from matplotlib import pyplot

try:  # matplotlib >= 3.5
    from matplotlib import colormaps as _colormaps

    def _get_cmap(name):
        return _colormaps[name]
except ImportError:  # older matplotlib
    from matplotlib import cm

    def _get_cmap(name):
        return cm.get_cmap(name)

min_samples = 256

with open("matplotlib_color_maps.py", "w") as file:
    file.write(
        """\"\"\"
matplotlib colormaps.

For more information, see
`Choosing Colormaps in Matplotlib <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.
\"\"\"
"""
    )

    for name in sorted(pyplot.colormaps()):
        cmap = _get_cmap(name)
        name_c = name.capitalize()
        if name_c == name:
            file.write(f"{name} = [ \n")
        else:
            # compability with older matplotlib_color_maps.py where all names were capitalized
            file.write(f"{name} = {name_c} = [ \n")

        # cmap.N is the actual number of datapoints the map is constructed with
        for x in np.linspace(0, 1, max(cmap.N, min_samples)):
            r, g, b = cmap(x)[:3]
            file.write("    {x:.4f}, {r:.4f}, {g:.4f}, {b:.4f},\n".format(**locals()))
        file.write("]\n\n")
