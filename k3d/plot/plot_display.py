import ipywidgets as widgets
from typing import Any

class PlotDisplayMixin:
    def display(self, **kwargs: Any) -> None:
        """Show plot inside ipywidgets.Output()."""
        output = widgets.Output()

        with output:
            display(self, **kwargs)

        self.outputs.append(output)

        display(output)

    def render(self) -> None:
        """Trigger rendering on demand.

        Useful when self.auto_rendering == False."""
        self.send({"msg_type": "render"})

    def start_auto_play(self) -> None:
        """Start animation of plot with objects using TimeSeries."""
        self.send({"msg_type": "start_auto_play"})

    def stop_auto_play(self) -> None:
        """Stop animation of plot with objects using TimeSeries."""
        self.send({"msg_type": "stop_auto_play"})

    def close(self) -> None:
        """Remove plot from all its ipywidgets.Output()-s."""
        for output in self.outputs:
            output.clear_output()

        self.outputs = [] 