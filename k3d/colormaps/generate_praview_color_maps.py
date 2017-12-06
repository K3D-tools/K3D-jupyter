import urllib.request
import json

response = urllib.request.urlopen('https://raw.githubusercontent.com/Kitware/ParaView/master/ParaViewCore/' +
                                  'ServerManager/Rendering/ColorMaps.json')
data = json.loads(response.read().decode('utf8'))

file = open('paraview_color_maps.py', 'w')

for item in data:
    if 'RGBPoints' in item:
        name = item['Name'].replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('2', 'two_')\
            .replace(',', '')
        name = name[:1].upper() + name[1:]
        file.write(name + ' = [\n')

        list_ = [item['RGBPoints'][i:i + 4] for i in range(0, len(item['RGBPoints']), 4)]

        for p in list_:
            file.write('        ' + str(p[0]) + ', ' + str(p[1]) + ', ' + str(p[2]) + ', ' + str(p[3]) + ',\n')

        file.write(']\n\n')

file.close()
