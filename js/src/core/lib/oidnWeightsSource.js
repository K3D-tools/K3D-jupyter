// The OIDN weights are binary files next to the bundle, not part of it: 1.8 MB each, read only
// once someone switches the denoiser on. Like the BVH worker, each entry point registers how to
// read a file of its own release - the widget asks the kernel, standalone fetches next to itself.
let read = null;
const cache = {};

module.exports = {
    provide(fn) {
        read = fn;
    },

    // resolves with the file as an ArrayBuffer, or null when this bundle cannot read it
    read(name) {
        if (!cache[name]) {
            cache[name] = read === null
                ? Promise.resolve(null)
                : Promise.resolve().then(() => read(name)).then((bytes) => bytes || null, () => null);
        }

        return cache[name];
    },
};
