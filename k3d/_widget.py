"""anywidget base class shared by every K3D widget."""

import pathlib

import anywidget
from traitlets import List, Unicode, default

_STATIC = pathlib.Path(__file__).parent / "static"

# ipywidgets/anywidget infrastructure traits (layout, tabbable, tooltip, ...) are
# transport machinery - they must not leak into scene state, snapshots or diffs
_BASE_TRAITS = set(anywidget.AnyWidget.class_trait_names())

# _esm rides in the synced state of EVERY widget instance, and K3D creates a widget
# per scene object - so objects and chunks carry this stub instead of the full module.
# It queues the model until the plot's widget.mjs loads and adopts the queue.
_MODEL_STUB = """
export default {
    initialize({ model }) {
        const REG = globalThis.__k3dWidgets = globalThis.__k3dWidgets || {};

        REG.pending = REG.pending || [];

        if (REG.adopt) {
            return REG.adopt(model);
        }

        const entry = { model, cancelled: false, adopted: false, cleanup: null };

        REG.pending.push(entry);

        return () => {
            entry.cancelled = true;
            if (entry.cleanup) {
                entry.cleanup();
            }
        };
    },
    render() {},
};
"""


class K3DAnyWidget(anywidget.AnyWidget):
    """One shared front-end module for all K3D widgets; the JS side dispatches on _kind.

    _synced_props lets the front end enumerate the synced traits - the anywidget model
    has no key listing, and the object/chunk widgets mirror their whole state into the
    scene on every change.
    """

    _esm = _STATIC / "widget.mjs"

    _kind = Unicode("").tag(sync=True)
    _synced_props = List(Unicode()).tag(sync=True)

    @default("_synced_props")
    def _default_synced_props(self):
        return sorted(
            name
            for name, trait in self.traits().items()
            if "sync" in trait.metadata
            and not name.startswith("_")
            and name not in _BASE_TRAITS
        )


class K3DModelWidget(K3DAnyWidget):
    """Model-only widgets (scene objects, voxel chunks): stub front end, no view."""

    _esm = _MODEL_STUB
