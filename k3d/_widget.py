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


_MODULE = _STATIC / "widget.mjs"

# Chunks of the module, served the same way for the same reason: the module runs from a blob
# URL, where nothing next to it has a resolvable URL. A fixed set - the front end must not be
# able to name any file it likes.
_ASSETS = {"k3d-bvh-worker.mjs"}

# _esm rides in the synced state of every instance, so the ~5 MB module is fetched from the
# kernel on demand and cached on globalThis: once per page, not once per plot.
_LOADER = """
const KEY = '__k3dWidgetModule';

function load(model) {
    if (globalThis[KEY]) {
        return globalThis[KEY];
    }

    globalThis[KEY] = new Promise((resolve, reject) => {
        function onMessage(msg, buffers) {
            if (!msg || msg.msg_type !== 'widget_module') {
                return;
            }

            model.off('msg:custom', onMessage);

            const buffer = buffers[0];
            const bytes = buffer instanceof Uint8Array
                ? buffer : new Uint8Array(buffer.buffer || buffer);
            const url = URL.createObjectURL(new Blob([bytes], { type: 'text/javascript' }));

            import(url).then((module) => {
                URL.revokeObjectURL(url);
                resolve(module.default);
            }, reject);
        }

        model.on('msg:custom', onMessage);
        model.send({ msg_type: 'fetch_widget_module' });
    }).catch((error) => {
        // one failed fetch must not leave every later widget on the page holding it
        delete globalThis[KEY];
        throw error;
    });

    return globalThis[KEY];
}

// Synchronous on purpose: anywidget gets a real cleanup function rather than a promise, and
// the module's own hooks still run in order, both waiting on the same promise.
function defer(hook) {
    return function (ctx) {
        let cleanup = null;
        let cancelled = false;

        load(ctx.model).then((module) => {
            if (!cancelled) {
                cleanup = module[hook](ctx);
            }
        }, (error) => {
            console.error('K3D: the widget module could not be loaded', error);
        });

        return () => {
            cancelled = true;

            if (typeof cleanup === 'function') {
                cleanup();
            }
        };
    };
}

export default { initialize: defer('initialize'), render: defer('render') };
"""


class K3DAnyWidget(anywidget.AnyWidget):
    """One shared front-end module for all K3D widgets; the JS side dispatches on _kind.

    _synced_props lets the front end enumerate the synced traits - the anywidget model
    has no key listing, and the object/chunk widgets mirror their whole state into the
    scene on every change.
    """

    _esm = _LOADER

    def _handle_custom_msg(self, content, buffers):
        if content.get("msg_type") == "fetch_widget_module":
            self.send({"msg_type": "widget_module"}, buffers=[_MODULE.read_bytes()])
            return

        if content.get("msg_type") == "fetch_widget_asset":
            name = content.get("name")
            asset = _STATIC / name if name in _ASSETS else None
            payload = [asset.read_bytes()] if asset is not None and asset.is_file() else []

            self.send({"msg_type": "widget_asset", "name": name}, buffers=payload)
            return

        super()._handle_custom_msg(content, buffers)

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
