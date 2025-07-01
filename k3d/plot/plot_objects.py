from ..objects import Drawable


class PlotObjectsMixin:
    def __iadd__(self, objs: Drawable) -> 'PlotObjectsMixin':
        """Add Drawable to plot."""
        assert isinstance(objs, Drawable)
        for obj in objs:
            if obj.id not in self.object_ids:
                self.object_ids = self.object_ids + [obj.id]
                self.objects.append(obj)
        return self

    def __isub__(self, objs: Drawable) -> 'PlotObjectsMixin':
        """Remove Drawable from plot."""
        assert isinstance(objs, Drawable)
        for obj in objs:
            self.object_ids = [id_ for id_ in self.object_ids if id_ != obj.id]
            if obj in self.objects:
                self.objects.remove(obj)
        return self
