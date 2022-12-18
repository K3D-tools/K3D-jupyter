import numpy as np
from matplotlib import pyplot, cm

min_samples = 256

with open('matplotlib_color_maps.py', 'w') as file:
    file.write("""\"\"\"
matplotlib colormaps.

For more information, see
`Choosing Colormaps in Matplotlib <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.
\"\"\"
""")

    for name in sorted(pyplot.colormaps()):
        cmap = cm.get_cmap(name)
        name_c = name.capitalize()
        if name_c == name:
            file.write('{} = [ \n'.format(name))
        else:
            # compability with older matplotlib_color_maps.py where all names were capitalized
            file.write('{} = {} = [ \n'.format(name, name_c))

        # cmap.N is the actual number of datapoints the map is constructed with
        for x in np.linspace(0, 1, max(cmap.N, min_samples)):
            r, g, b = cmap(x)[:3]
            file.write('    {x:.4f}, {r:.4f}, {g:.4f}, {b:.4f},\n'.format(**locals()))
        file.write(']\n\n')
