"""anywidget base class shared by every K3D widget."""

import pathlib

import anywidget
from traitlets import List, Unicode

_STATIC = pathlib.Path(__file__).parent / "static"

# ipywidgets/anywidget infrastructure traits (layout, tabbable, tooltip, ...) are
# transport machinery - they must not leak into scene state, snapshots or diffs
_BASE_TRAITS = set(anywidget.AnyWidget.class_trait_names())


class K3DAnyWidget(anywidget.AnyWidget):
    """One shared front-end module for all K3D widgets; the JS side dispatches on _kind.

    _synced_props lets the front end enumerate the synced traits - the anywidget model
    has no key listing, and the object/chunk widgets mirror their whole state into the
    scene on every change.
    """

    _esm = _STATIC / "widget.mjs"

    _kind = Unicode("").tag(sync=True)
    _synced_props = List(Unicode()).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._synced_props = sorted(
            name
            for name, trait in self.traits().items()
            if "sync" in trait.metadata
            and not name.startswith("_")
            and name not in _BASE_TRAITS
        )
