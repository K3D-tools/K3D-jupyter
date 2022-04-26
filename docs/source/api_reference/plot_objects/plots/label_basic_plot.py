import k3d


def generate():
    plt_label1 = k3d.label('Insert text here',
                           position=(1, 1, 1))
    plt_label2 = k3d.label('Insert text here (HTML)',
                           position=(-1, -1, -1),
                           is_html=True)

    plot = k3d.plot()
    plot += plt_label1
    plot += plt_label2

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
