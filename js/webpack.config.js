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
    { test: /\.css$/, use: ['style-loader', 'css-loader'] },
];

const mode = 'production';

// lil-gui 0.21 added an exports field whose `require` condition serves the UMD build, and its
// anonymous define() breaks the AMD loader standalone snapshots run on.
const resolve = {
    alias: {
        'lil-gui': path.resolve(__dirname, 'node_modules/lil-gui/dist/lil-gui.esm.js'),
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
            path: `${__dirname}/../k3d/static`,
            library: {
                type: 'module',
            },
            publicPath: '',
        },
        mode,
        devtool: 'source-map',
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
                path: `${__dirname}/../k3d/static`,
                library: 'k3d',
                libraryTarget: 'amd',
                publicPath: `https://unpkg.com/k3d@${version}/dist/`,
            },
        mode,
        devtool: 'source-map',
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
                    compiler.hooks.afterEmit.tap('CopyBuildPlugin', (compilation) => {
                        const outputPath = compiler.options.output.path;
                        const files = ['standalone.js', 'standalone.js.map'];
                        const targetDir = path.resolve(__dirname, 'dist');

                        if (!fs.existsSync(targetDir)) {
                            fs.mkdirSync(targetDir, { recursive: true });
                        }

                        files.forEach(file => {
                            const src = path.join(outputPath, file);
                            if (fs.existsSync(src)) {
                                fs.copyFileSync(src, path.join(targetDir, file));
                            }
                        });
                    });
                }
            }
        ],
    },
];
