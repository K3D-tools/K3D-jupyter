import k3d


def generate():
    plt_text1 = k3d.text('Insert text here',
                         position=(1, 1, 1))
    plt_text2 = k3d.text('Insert text here (HTML)',
                         position=(-1, -1, -1),
                         is_html=True)

    plot = k3d.plot()
    plot += plt_text1
    plot += plt_text2

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
