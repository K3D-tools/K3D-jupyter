import numpy as np
from matplotlib import pyplot, cm


min_samples = 256
with open('matplotlib_color_maps.py', 'w') as file:
    file.write('import numpy as np\n\n')

    for name in sorted(pyplot.colormaps()):
        cmap = cm.get_cmap(name)
        name_c = name.capitalize()
        if name_c == name:
            file.write('{} = np.array([ \n'.format(name))
        else:
            # compability with older matplotlib_color_maps.py where all names were capitalized
            file.write('{} = {} = np.array([ \n'.format(name, name_c))

        # cmap.N is the actual number of datapoints the map is constructed with
        for x in np.linspace(0, 1, max(cmap.N, min_samples)):
            r, g, b = cmap(x)[:3]
            file.write('    {x}, {r}, {g}, {b},\n'.format(**locals()))
        file.write('], dtype=np.float32)\n\n')
