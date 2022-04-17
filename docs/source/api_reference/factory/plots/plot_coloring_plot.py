import k3d


def generate():
    plot = k3d.plot(background_color=0x1e1e1e,
                    grid_color=0xd2d2d2,
                    label_color=0xf0f0f0)

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
