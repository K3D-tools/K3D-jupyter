from typing import List

from ..objects.base import TimeSeries


class PlotTimeMixin:
    """Time series introspection and frame stepping, answered without a browser."""

    def get_time_series_times(self) -> List[float]:
        """Sorted keyframe times of every time series in the plot, deduplicated.

        Objects need not share keys, so the result is their union. camera_animation counts too.
        """
        times = set()

        for obj in self.objects:
            for name, trait in obj.traits().items():
                if not isinstance(trait, TimeSeries):
                    continue

                value = obj[name]

                if not isinstance(value, dict):
                    continue

                for key in value.keys():
                    try:
                        times.add(float(key))
                    except (TypeError, ValueError):
                        continue

        # Only a dict carries keyframes; the trait defaults to a list.
        if isinstance(self.camera_animation, dict):
            for key in self.camera_animation.keys():
                try:
                    times.add(float(key))
                except (TypeError, ValueError):
                    continue

        return sorted(times)

    def get_time_series_range(self) -> List[float]:
        """[min, max] of the keyframe times, always including 0.0, which is what time clamps to."""
        times = self.get_time_series_times()

        return [min([0.0] + times), max([0.0] + times)]

    def step_frame(self, step: int = 1) -> float:
        """Move `time` by whole keyframes and return the new value.

        Steps from the keyframe nearest the current time and clamps at both ends; it does not wrap.
        """
        times = self.get_time_series_times()

        if not times:
            return self.time

        nearest = min(range(len(times)), key=lambda i: abs(times[i] - self.time))
        index = min(max(nearest + step, 0), len(times) - 1)

        self.time = times[index]

        return self.time

    def next_frame(self) -> float:
        """Move `time` to the next keyframe and return the new value."""
        return self.step_frame(1)

    def previous_frame(self) -> float:
        """Move `time` to the previous keyframe and return the new value."""
        return self.step_frame(-1)
