import urllib.request
import xml.etree.ElementTree as ET

response = urllib.request.urlopen('http://www.paraview.org/Wiki/images/d/d4/All_mpl_cmaps.xml')
root = ET.fromstring(response.read().decode('utf8'))

file = open('matplotlib_color_maps.py', 'w')
file.write('class matplotlib_color_maps():\n\n')

for colorMap in root:
    name = colorMap.attrib['name'].capitalize()
    file.write('    ' + name + ' = [\n')

    for p in colorMap:
        file.write('        ' + p.attrib['x'] + ', ' + p.attrib['r'] + ', ' + p.attrib['g'] + ', '
                   + p.attrib['b'] + ',\n')

    file.write('    ]\n\n')

file.close()
