from typing import Any

import ipywidgets as widgets
from IPython.display import display


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

        Useful when self.render_on_change == False."""
        self.send({"msg_type": "render"})

    def start_auto_play(self) -> None:
        """Start animation of plot with objects using TimeSeries."""
        self.send({"msg_type": "start_auto_play"})

    def stop_auto_play(self) -> None:
        """Stop animation of plot with objects using TimeSeries."""
        self.send({"msg_type": "stop_auto_play"})

    def close(self) -> None:
        """Remove plot from all its ipywidgets.Output()-s."""
        # Widget.__del__ calls this, and it fires on a plot whose __init__ raised - a rejected
        # trait then reports itself twice, the second time as an unrelated AttributeError
        for output in getattr(self, "outputs", []):
            output.clear_output()

        self.outputs = []
