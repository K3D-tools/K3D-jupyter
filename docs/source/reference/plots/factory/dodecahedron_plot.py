import k3d

platonic = k3d.platonic


def generate():
    plot = k3d.plot()

    dodec_1 = platonic.Dodecahedron()
    dodec_2 = platonic.Dodecahedron(origin=[5, -2, 3], size=0.5)

    plot += dodec_1.mesh
    plot += dodec_2.mesh

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
