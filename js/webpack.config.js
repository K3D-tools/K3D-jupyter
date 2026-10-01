const CopyPlugin = require('copy-webpack-plugin');
const path = require('path');
const fs = require('fs');
const version = require('./package.json').version;

// Custom webpack loaders are generally the same for all webpack bundles, hence
// stored in a separate local variable.
const rules = [
    {
        test: /\.(png|jpg|gif|svg|eot|ttf|woff|woff2)$/,
        type: 'asset/inline',
    },
    {
        test: /\.(glsl|txt)$/,
        type: 'asset/source',
    },
    {
        resourceQuery: /raw/,
        type: 'asset/source',
    },
    {
        test: /\.css$/,
        use: [
            'style-loader',
            {
                loader: 'css-loader',
                // KaTeX ships each font as woff2/woff/ttf and all three would be inlined; woff2
                // alone covers every browser that has WebGL2, and it is first in every src list
                options: { url: { filter: (url) => !/\.(eot|ttf|woff)(\?.*)?$/.test(url) } },
            },
        ],
    },
];

const mode = 'production';

// lil-gui 0.21 added an exports field whose `require` condition serves the UMD build, and its
// anonymous define() breaks the AMD loader standalone snapshots run on.
const resolve = {
    alias: {
        'lil-gui': path.resolve(__dirname, 'node_modules/lil-gui/dist/lil-gui.esm.js'),
        // js/src is CommonJS and three-gpu-pathtracer is ESM, so webpack resolved both export
        // conditions and shipped two complete, mutually incompatible copies of three.js - and of
        // three-mesh-bvh behind it. Two class hierarchies means every instanceof between the
        // raster path and the cinematic one is a coin toss, and users get a console warning
        // about it. Same reason as lil-gui above: name the build, end the ambiguity.
        three$: path.resolve(__dirname, 'node_modules/three/build/three.module.js'),
        'three-mesh-bvh$': path.resolve(__dirname, 'node_modules/three-mesh-bvh/src/index.js'),
    },
};

module.exports = [
    { // anywidget front-end module - the whole Jupyter/Colab/VS Code widget layer
        entry: './src/anywidget.js',
        experiments: {
            outputModule: true,
        },
        output: {
            filename: 'widget.mjs',
            // the BVH worker is the one chunk; a fixed name is what lets the module ask the
            // kernel for it, and a second chunk would collide loudly here rather than silently
            chunkFilename: 'k3d-bvh-worker.mjs',
            path: `${__dirname}/../k3d/static`,
            library: {
                type: 'module',
            },
            publicPath: '',
        },
        mode,
        // hidden: the map is emitted for local debugging but the bundle stops pointing at it.
        // widget.mjs reaches the browser through the Jupyter comm, not from a URL, so a
        // sourceMappingURL in it can never resolve - every user who opens devtools on a plot
        // gets a 404 for a file the wheel does not even carry.
        devtool: 'hidden-source-map',
        resolve,
        module: {
            rules,
        },
    },
    { // standalone bundle - snapshots (full/online/inline), headless, docs
        entry: './src/standalone.js',
        output:
            {
                filename: 'standalone.js',
                chunkFilename: 'k3d-bvh-worker.js',
                path: `${__dirname}/../k3d/static`,
                library: 'k3d',
                libraryTarget: 'amd',
                publicPath: `https://unpkg.com/k3d@${version}/dist/`,
            },
        mode,
        // hidden as well, and the map is no longer published: it was 9.8 MB of a 13.9 MB npm
        // package - 71% - for a consumer nobody could name, and the worker chunk's own map was
        // never copied at all, so unpkg served a dangling reference beside it.
        devtool: 'hidden-source-map',
        resolve,
        module: {
            rules,
        },
        plugins: [
            new CopyPlugin({
                patterns: [
                    { from: './src/core/lib/headless.html' },
                    { from: './src/core/lib/snapshot_standalone.txt' },
                    { from: './src/core/lib/snapshot_online.txt' },
                    { from: './src/core/lib/snapshot_inline.txt' },
                    { from: './node_modules/requirejs/require.js' },
                    { from: './node_modules/fflate/umd/index.js', to: 'fflate.js' },
                ],
            }),
            // js/dist mirrors what npm publishes (unpkg serves standalone for the
            // online/inline snapshot templates) and what docs/source/conf.py copies
            {
                apply: (compiler) => {
                    compiler.hooks.afterEmit.tap('CopyBuildPlugin', () => {
                        const outputPath = compiler.options.output.path;
                        const files = ['standalone.js', 'k3d-bvh-worker.js'];
                        const targetDir = path.resolve(__dirname, 'dist');

                        if (!fs.existsSync(targetDir)) {
                            fs.mkdirSync(targetDir, { recursive: true });
                        }

                        files.forEach((file) => {
                            const src = path.join(outputPath, file);
                            if (fs.existsSync(src)) {
                                fs.copyFileSync(src, path.join(targetDir, file));
                            }
                        });
                    });
                },
            },
        ],
    },
];
