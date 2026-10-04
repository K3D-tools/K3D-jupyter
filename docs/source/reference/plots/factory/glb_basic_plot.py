import k3d
from k3d.helpers import download


def generate():
    # Model: Cesium Milk Truck, (c) 2017 Cesium, CC-BY 4.0,
    # https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMilkTruck
    filename = download('https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/'
                        'main/Models/CesiumMilkTruck/glTF-Binary/CesiumMilkTruck.glb')

    truck = k3d.glb(filename)

    plot = k3d.plot(renderer='advanced')
    plot += truck

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
