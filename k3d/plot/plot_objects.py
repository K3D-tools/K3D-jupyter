from ..objects import Drawable


def _assert_drawable(objs, operator):
    if isinstance(objs, Drawable):
        return

    if isinstance(objs, (list, tuple, set)):
        raise TypeError(
            "plot %s takes one Drawable, not a %s - combine objects with +: plot %s a + b"
            % (operator, type(objs).__name__, operator)
        )

    raise TypeError("plot %s takes a Drawable, not %s" % (operator, type(objs).__name__))


class PlotObjectsMixin:
    def __iadd__(self, objs: Drawable) -> "PlotObjectsMixin":
        """Add Drawable to plot."""
        _assert_drawable(objs, "+=")
        for obj in objs:
            if obj.id not in self.object_ids:
                self.object_ids = self.object_ids + [obj.id]
                self.objects.append(obj)
        return self

    def __isub__(self, objs: Drawable) -> "PlotObjectsMixin":
        """Remove Drawable from plot."""
        _assert_drawable(objs, "-=")
        for obj in objs:
            self.object_ids = [id_ for id_ in self.object_ids if id_ != obj.id]
            if obj in self.objects:
                self.objects.remove(obj)
        return self
