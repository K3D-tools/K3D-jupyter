import k3d


def generate():
    plt_points = k3d.points([[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                            point_size=0.5,
                            shader='flat',
                            colors=[0xff0000, 0x00ff00, 0x0000ff])

    plt_label_dynamic = k3d.label('Dynamic',
                                  position=(1, 1, 1),
                                  mode='dynamic',
                                  label_box=False,
                                  color=0xff0000)
    plt_label_local = k3d.label('Local',
                                position=(0, 0, 0),
                                mode='local',
                                label_box=False,
                                color=0x00ff00)
    plt_label_side = k3d.label('Side',
                               position=(-1, -1, -1),
                               mode='side',
                               label_box=False,
                               color=0x0000ff)

    plot = k3d.plot()
    plot += plt_points
    plot += plt_label_dynamic
    plot += plt_label_local
    plot += plt_label_side

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
