class PlotSerializationMixin:
    def get_static_path(self) -> str:
        import os

        dir_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(dir_path, "../static")

    def __getstate__(self) -> bytes:
        return self.get_binary_snapshot()

    def __setstate__(self, data: bytes) -> None:
        self.__init__()
        self.load_binary_snapshot(data)
