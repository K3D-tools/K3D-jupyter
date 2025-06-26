import json
import urllib.request

response = urllib.request.urlopen(
    'https://gitlab.kitware.com/paraview/paraview/raw/master/Remoting/Views/ColorMaps.json')
data = json.loads(response.read().decode('utf8'))

file = open('paraview_color_maps.py', 'w')

file.write("""\"\"\"
ParaView colormaps.
\"\"\"
""")

for item in data:
    if 'RGBPoints' in item:
        name = item['Name'].replace(' ', '_').replace('-', '_').replace('(', '').replace(')',
                                                                                         '').replace(
            '2', 'two_') \
            .replace(',', '')
        name = name[:1].upper() + name[1:]
        file.write(name + ' = [\n')

        list_ = [item['RGBPoints'][i:i + 4] for i in range(0, len(item['RGBPoints']), 4)]

        for p in list_:
            x, r, g, b = p
            file.write('    {x:.4f}, {r:.4f}, {g:.4f}, {b:.4f},\n'.format(**locals()))

        file.write(']\n\n')

file.close()
