// The BVH worker script is a webpack chunk, and neither bundle can construct it the way
// webpack does - relative to the module. The widget module is imported from a blob URL, where
// resolving anything relative throws, and standalone resolves against unpkg, which is another
// origin. So each entry point registers a way to read the chunk's source instead, and the
// cinematic backend runs it as a blob worker.
//
// The standalone chunk carries an AMD wrapper: executing it needs a define() that just calls
// the factory. Harmless in front of the plain ESM chunk the widget bundle emits.
const DEFINE_SHIM = 'var define=function(n,d,f){f();};';

let read = null;

module.exports = {
    provide(fn) {
        read = fn;
    },

    // resolves with runnable worker source, or null when this bundle cannot read it - the
    // caller's fallback is the synchronous build, so a failure here is not an error
    read() {
        if (read === null) {
            return Promise.resolve(null);
        }

        return Promise.resolve().then(read).then(
            (source) => (source ? DEFINE_SHIM + source : null),
            () => null,
        );
    },
};
