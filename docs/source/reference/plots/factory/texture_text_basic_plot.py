import k3d


def generate():
    plt_texture_text = k3d.texture_text('Texture',
                                        position=[0, 0, 0],
                                        font_face='Calibri',
                                        font_weight=600,
                                        color=0xa2ffc8)

    plot = k3d.plot()
    plot += plt_texture_text

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
