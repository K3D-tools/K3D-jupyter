import k3d


def generate():
    plot = k3d.plot(camera_no_pan=True,
                    camera_no_rotate=True,
                    camera_no_zoom=True,
                    menu_visibility=False)

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
