def _unpickle_plot(data: bytes):
    from . import Plot

    plot = Plot()
    plot.load_binary_snapshot(data)

    return plot


class PlotSerializationMixin:
    # anywidget's add_traits swaps the instance's class for a dynamic subclass that pickle
    # cannot find by name; a module-level function can be found, and it rebuilds from the
    # same binary snapshot __getstate__ already produced
    def __reduce__(self):
        return (_unpickle_plot, (self.get_binary_snapshot(),))

    def get_static_path(self) -> str:
        import os

        dir_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(dir_path, "../static")

    def __getstate__(self) -> bytes:
        return self.get_binary_snapshot()

    def __setstate__(self, data: bytes) -> None:
        self.__init__()
        self.load_binary_snapshot(data)
